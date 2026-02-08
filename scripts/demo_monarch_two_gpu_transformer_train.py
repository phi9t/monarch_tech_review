#!/usr/bin/env python3
"""Train a tiny Qwen-style transformer on two GPUs using Monarch actors."""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from transformers import Qwen2Config, Qwen2ForCausalLM


@dataclass
class TrainStats:
    rank: int
    device: int
    steps: int
    start_loss: float
    end_loss: float
    avg_loss: float
    best_loss: float
    tokens_per_sec: float
    samples_per_sec: float
    step_time_ms: float
    peak_mem_mb: float


@dataclass
class RankProgress:
    rank: int
    device: int
    processed_batches: int = 0
    last_loss: float = 0.0
    avg_loss: float = 0.0
    best_loss: float = math.inf
    avg_step_ms: float = 0.0
    tokens_per_sec: float = 0.0


@dataclass
class DataPlaneStats:
    total_rounds: int = 0
    rounds_processed: int = 0
    total_rank_batches: int = 0
    processed_rank_batches: int = 0
    remaining_rank_batches: int = 0
    pct_complete: float = 0.0
    elapsed_s: float = 0.0
    avg_batch_time_ms: float = 0.0
    throughput_batches_per_s: float = 0.0
    throughput_tokens_per_s: float = 0.0
    loss_last: float = 0.0
    loss_avg: float = 0.0
    loss_best: float = math.inf
    eta_s: float = 0.0
    active_ranks: int = 0
    done_ranks: int = 0


class Dashboard:
    def __init__(self, console: Console, args: argparse.Namespace):
        self.console = console
        self.args = args
        self.start_time = time.perf_counter()
        self.control_state = "init"

        self.control_lines: deque[str] = deque(maxlen=18)
        self.gpu_lines: deque[str] = deque(maxlen=18)
        self.log_lines: deque[str] = deque(maxlen=args.log_max_lines)

        self.data_stats = DataPlaneStats()
        self.rank_stats: dict[int, RankProgress] = {}
        self.spinner_idx = 0

        self.last_gpu_sample_ts = 0.0
        self.last_gpu_error = ""

    def control(self, message: str) -> None:
        self.control_lines.append(message)
        self.log(f"[CTRL] {message}")

    def set_control_state(self, state: str, message: str) -> None:
        self.control_state = state
        self.control(message)

    def log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{ts}] {message}")
        if not self.console.is_terminal:
            self.console.print(f"[{ts}] {message}")

    def init_data_stats(self, total_rounds: int, num_ranks: int) -> None:
        self.data_stats = DataPlaneStats(
            total_rounds=total_rounds,
            rounds_processed=0,
            total_rank_batches=total_rounds * num_ranks,
            processed_rank_batches=0,
            remaining_rank_batches=total_rounds * num_ranks,
            active_ranks=num_ranks,
            done_ranks=0,
            loss_best=math.inf,
        )
        self.rank_stats = {}

    def update_data_from_step(self, round_idx: int, step_results: dict[Any, dict], batch_size: int, seq_len: int) -> None:
        now = time.perf_counter()
        elapsed = max(now - self.start_time, 1e-6)
        losses = []
        per_step_ms = []
        total_step_tokens_per_s = 0.0

        for _, result in step_results.items():
            rank = int(result["rank"])
            if rank not in self.rank_stats:
                self.rank_stats[rank] = RankProgress(rank=rank, device=int(result["device"]))
            r = self.rank_stats[rank]
            r.processed_batches = int(result["step"])
            r.last_loss = float(result["loss"])
            r.avg_loss = float(result["avg_loss"])
            r.best_loss = min(r.best_loss, float(result["loss"]))
            r.avg_step_ms = float(result["avg_step_ms"])
            r.tokens_per_sec = float(result["tokens_per_sec"])

            losses.append(float(result["loss"]))
            per_step_ms.append(float(result["step_time_ms"]))
            total_step_tokens_per_s += float(result["tokens_per_sec"])

        num_ranks = max(len(step_results), 1)
        processed_rank_batches = round_idx * num_ranks
        remaining_rank_batches = max(self.data_stats.total_rank_batches - processed_rank_batches, 0)
        pct_complete = (processed_rank_batches / max(self.data_stats.total_rank_batches, 1)) * 100.0
        throughput_batches_per_s = processed_rank_batches / elapsed
        throughput_tokens_per_s = throughput_batches_per_s * batch_size * seq_len
        avg_batch_time_ms = (sum(per_step_ms) / max(len(per_step_ms), 1))
        avg_rank_batch_time_s = avg_batch_time_ms / 1000.0
        eta_s = remaining_rank_batches * avg_rank_batch_time_s

        self.data_stats.rounds_processed = round_idx
        self.data_stats.processed_rank_batches = processed_rank_batches
        self.data_stats.remaining_rank_batches = remaining_rank_batches
        self.data_stats.pct_complete = pct_complete
        self.data_stats.elapsed_s = elapsed
        self.data_stats.avg_batch_time_ms = avg_batch_time_ms
        self.data_stats.throughput_batches_per_s = throughput_batches_per_s
        self.data_stats.throughput_tokens_per_s = throughput_tokens_per_s
        self.data_stats.loss_last = sum(losses) / max(len(losses), 1)
        self.data_stats.loss_avg = (
            sum(r.avg_loss for r in self.rank_stats.values()) / max(len(self.rank_stats), 1)
        )
        self.data_stats.loss_best = min((r.best_loss for r in self.rank_stats.values()), default=math.inf)
        self.data_stats.eta_s = eta_s
        self.data_stats.active_ranks = len(self.rank_stats)
        self.data_stats.done_ranks = 0

    def finalize_data(self, final_results: dict[Any, dict], batch_size: int, seq_len: int) -> None:
        rounds = 0
        num_ranks = len(final_results)
        total_tokens_per_s = 0.0
        losses = []
        avg_step = []
        for _, result in final_results.items():
            rank = int(result["rank"])
            rounds = max(rounds, int(result["steps"]))
            if rank not in self.rank_stats:
                self.rank_stats[rank] = RankProgress(rank=rank, device=int(result["device"]))
            r = self.rank_stats[rank]
            r.processed_batches = int(result["steps"])
            r.last_loss = float(result["end_loss"])
            r.avg_loss = float(result["avg_loss"])
            r.best_loss = float(result["best_loss"])
            r.avg_step_ms = float(result["step_time_ms"])
            r.tokens_per_sec = float(result["tokens_per_sec"])

            total_tokens_per_s += float(result["tokens_per_sec"])
            losses.append(float(result["end_loss"]))
            avg_step.append(float(result["step_time_ms"]))

        self.data_stats.rounds_processed = rounds
        self.data_stats.processed_rank_batches = rounds * num_ranks
        self.data_stats.remaining_rank_batches = 0
        self.data_stats.pct_complete = 100.0
        self.data_stats.avg_batch_time_ms = sum(avg_step) / max(len(avg_step), 1)
        self.data_stats.throughput_batches_per_s = (
            total_tokens_per_s / max(batch_size * seq_len, 1)
        )
        self.data_stats.throughput_tokens_per_s = total_tokens_per_s
        self.data_stats.loss_last = sum(losses) / max(len(losses), 1)
        self.data_stats.loss_avg = (
            sum(r.avg_loss for r in self.rank_stats.values()) / max(len(self.rank_stats), 1)
        )
        self.data_stats.loss_best = min((r.best_loss for r in self.rank_stats.values()), default=math.inf)
        self.data_stats.eta_s = 0.0
        self.data_stats.done_ranks = num_ranks

    def sample_gpu_stats(self) -> None:
        self.last_gpu_sample_ts = time.perf_counter()
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if not lines:
                raise RuntimeError("nvidia-smi returned no data")

            self.gpu_lines.clear()
            for line in lines:
                idx, name, util, mem_used, mem_total, temp = [p.strip() for p in line.split(",")]
                mem_pct = (float(mem_used) / max(float(mem_total), 1.0)) * 100.0
                util_bar = "█" * max(1, min(10, int(round(float(util) / 10.0))))
                self.gpu_lines.append(
                    f"GPU{idx} {name[:16]:<16} util {float(util):>5.1f}% {util_bar:<10} "
                    f"mem {float(mem_used):>5.0f}/{float(mem_total):.0f}MB ({mem_pct:>5.1f}%) "
                    f"temp {float(temp):>4.0f}C"
                )
            self.last_gpu_error = ""
        except Exception as e:
            self.gpu_lines.clear()
            self.last_gpu_error = str(e)
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    used = total - free
                    self.gpu_lines.append(
                        f"GPU{i} fallback mem used={used/1024**2:,.0f}MB total={total/1024**2:,.0f}MB"
                    )
                except Exception as sub_e:
                    self.gpu_lines.append(f"GPU{i} fallback unavailable: {sub_e}")
            self.log(f"[GPU] nvidia-smi unavailable, using torch fallback: {e}")

    def _control_panel(self) -> Panel:
        elapsed = time.perf_counter() - self.start_time
        head = [
            f"state={self.control_state}",
            f"elapsed={elapsed:.1f}s",
            f"steps={self.args.steps} batch={self.args.batch_size} seq={self.args.seq_len}",
        ]
        body = "\n".join(head + list(self.control_lines))
        return Panel(Text(body, overflow="fold"), title="Control Plane", border_style="cyan")

    def _data_panel(self) -> Panel:
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = (self.spinner_idx + 1) % len(spinner)
        ds = self.data_stats
        lines = [
            f"{spinner[self.spinner_idx]} Data-path status: active={ds.active_ranks} done={ds.done_ranks}",
            f"Rounds: {ds.rounds_processed}/{ds.total_rounds}",
            (
                f"Rank-batches: {ds.processed_rank_batches}/{ds.total_rank_batches} "
                f"({ds.pct_complete:.1f}%) remaining={ds.remaining_rank_batches}"
            ),
            (
                f"Timing: elapsed={ds.elapsed_s:.1f}s avg_batch={ds.avg_batch_time_ms:.1f}ms "
                f"eta={ds.eta_s:.1f}s"
            ),
            (
                f"Throughput: {ds.throughput_batches_per_s:.2f} batch/s "
                f"{ds.throughput_tokens_per_s:.1f} tok/s"
            ),
            f"Loss: last={ds.loss_last:.4f} avg={ds.loss_avg:.4f} best={ds.loss_best:.4f}",
            "-" * 48,
            "Per-rank: rank gpu processed last_loss avg_loss avg_ms tok/s",
        ]
        for rank in sorted(self.rank_stats):
            rs = self.rank_stats[rank]
            lines.append(
                f"R{rs.rank:>1}   G{rs.device:>1}   {rs.processed_batches:>3}      "
                f"{rs.last_loss:>7.4f}  {rs.avg_loss:>7.4f}  {rs.avg_step_ms:>6.1f}  {rs.tokens_per_sec:>7.1f}"
            )
        return Panel(Text("\n".join(lines), overflow="fold"), title="Data Plane (Stats)", border_style="green")

    def _gpu_panel(self) -> Panel:
        lines = list(self.gpu_lines) if self.gpu_lines else ["No GPU telemetry yet"]
        footer = f"sample_age={max(0.0, time.perf_counter() - self.last_gpu_sample_ts):.1f}s"
        if self.last_gpu_error:
            footer += f" | fallback active: {self.last_gpu_error[:60]}"
        return Panel(
            Text("\n".join(lines), overflow="fold"),
            title=f"GPU Usage (sample {time.strftime('%H:%M:%S')})",
            border_style="yellow",
            subtitle=footer,
        )

    def _logs_panel(self) -> Panel:
        tail = list(self.log_lines)[-max(self.args.log_tail_lines, 1) :]
        logs = "\n".join(tail) if tail else "No logs yet"
        return Panel(Text(logs, overflow="fold"), title="Program Logs (tail)", border_style="magenta")

    def render(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(Layout(name="top", size=16), Layout(name="logs"))
        layout["top"].split_row(Layout(name="control"), Layout(name="data"), Layout(name="gpu"))
        layout["control"].update(self._control_panel())
        layout["data"].update(self._data_panel())
        layout["gpu"].update(self._gpu_panel())
        layout["logs"].update(self._logs_panel())
        return layout


class TinyQwenTrainer(Actor):
    def __init__(self):
        self._is_setup = False
        self._rank = -1
        self._device = -1
        self._model = None
        self._optimizer = None
        self._config = None
        self._steps = 0
        self._total_loss = 0.0
        self._best_loss = math.inf
        self._start_loss = None
        self._end_loss = 0.0
        self._start_t = 0.0
        self._peak_mem_mb = 0.0
        self._batch_size = 0
        self._seq_len = 0

    @endpoint
    def setup(self, batch_size: int, seq_len: int, lr: float) -> dict:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this demo")

        self._rank = current_rank().rank
        self._device = self._rank % torch.cuda.device_count()
        torch.cuda.set_device(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)

        self._config = Qwen2Config(
            vocab_size=32000,
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=max(seq_len, 256),
        )
        self._model = Qwen2ForCausalLM(self._config).to(self._device)
        self._model.train()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)

        self._steps = 0
        self._total_loss = 0.0
        self._best_loss = math.inf
        self._start_loss = None
        self._end_loss = 0.0
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._start_t = time.perf_counter()
        self._is_setup = True

        return {"rank": self._rank, "device": self._device}

    @endpoint
    def train_step(self, batch_size: int, seq_len: int) -> dict:
        if not self._is_setup or self._model is None or self._optimizer is None or self._config is None:
            raise RuntimeError("Trainer not initialized; call setup() first")

        t0 = time.perf_counter()
        input_ids = torch.randint(
            low=0,
            high=self._config.vocab_size,
            size=(batch_size, seq_len),
            device=self._device,
            dtype=torch.long,
        )
        outputs = self._model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss

        if self._start_loss is None:
            self._start_loss = float(loss.item())
        self._end_loss = float(loss.item())
        self._total_loss += self._end_loss
        self._best_loss = min(self._best_loss, self._end_loss)

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._optimizer.step()
        torch.cuda.synchronize(self._device)
        step_elapsed = max(time.perf_counter() - t0, 1e-6)

        self._steps += 1
        elapsed = max(time.perf_counter() - self._start_t, 1e-6)
        tokens = batch_size * seq_len
        self._peak_mem_mb = max(
            self._peak_mem_mb,
            torch.cuda.max_memory_allocated(self._device) / (1024**2),
        )
        return {
            "rank": self._rank,
            "device": self._device,
            "step": self._steps,
            "loss": self._end_loss,
            "avg_loss": self._total_loss / max(self._steps, 1),
            "best_loss": self._best_loss,
            "step_time_ms": step_elapsed * 1000.0,
            "avg_step_ms": (elapsed / max(self._steps, 1)) * 1000.0,
            "tokens_per_sec": tokens / step_elapsed,
            "peak_mem_mb": self._peak_mem_mb,
        }

    @endpoint
    def finalize(self) -> dict:
        elapsed = max(time.perf_counter() - self._start_t, 1e-6)
        tokens = self._steps * self._batch_size * self._seq_len
        stats = TrainStats(
            rank=self._rank,
            device=self._device,
            steps=self._steps,
            start_loss=float(self._start_loss if self._start_loss is not None else self._end_loss),
            end_loss=float(self._end_loss),
            avg_loss=self._total_loss / max(self._steps, 1),
            best_loss=float(self._best_loss),
            tokens_per_sec=tokens / elapsed,
            samples_per_sec=(self._steps * self._batch_size) / elapsed,
            step_time_ms=(elapsed / max(self._steps, 1)) * 1000.0,
            peak_mem_mb=self._peak_mem_mb,
        )
        return asdict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--refresh-hz", type=float, default=4.0)
    parser.add_argument("--log-max-lines", type=int, default=400)
    parser.add_argument("--log-tail-lines", type=int, default=40)
    parser.add_argument("--metrics-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    dashboard = Dashboard(console=console, args=args)

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(f"Need >=2 CUDA GPUs, found {gpu_count}")

    mesh_results: dict[Any, dict] | None = None

    with Live(
        dashboard.render(),
        console=console,
        refresh_per_second=max(args.refresh_hz, 1.0),
        screen=console.is_terminal,
        transient=False,
    ) as live:
        dashboard.set_control_state("init", f"Detected {gpu_count} CUDA GPUs")
        dashboard.set_control_state("mesh_create", "Creating HostMesh via this_host()")
        dashboard.sample_gpu_stats()
        live.update(dashboard.render())
        host = this_host()

        dashboard.set_control_state("proc_spawn", "Spawning ProcMesh with per_host={'gpus': 2}")
        procs = host.spawn_procs(per_host={"gpus": 2})
        dashboard.sample_gpu_stats()
        live.update(dashboard.render())

        try:
            dashboard.set_control_state("actor_spawn", "Spawning TinyQwenTrainer actor mesh")
            trainers = procs.spawn("tiny_qwen_trainers", TinyQwenTrainer)
            live.update(dashboard.render())

            dashboard.set_control_state("setup", "Initializing trainer actors")
            setup_mesh = trainers.setup.call(args.batch_size, args.seq_len, args.lr).get()
            num_ranks = len(setup_mesh)
            dashboard.init_data_stats(total_rounds=args.steps, num_ranks=num_ranks)
            dashboard.log(f"[DATA] initialized {num_ranks} trainer ranks")
            dashboard.sample_gpu_stats()
            live.update(dashboard.render())

            dashboard.set_control_state("train_loop", "Running synchronous training rounds")
            for round_idx in range(1, args.steps + 1):
                step_mesh = trainers.train_step.call(args.batch_size, args.seq_len).get()
                dashboard.update_data_from_step(
                    round_idx=round_idx,
                    step_results=step_mesh,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                )
                if round_idx % max(args.metrics_every, 1) == 0:
                    dashboard.log(
                        f"[DATA] round={round_idx}/{args.steps} "
                        f"processed={dashboard.data_stats.processed_rank_batches}/"
                        f"{dashboard.data_stats.total_rank_batches} "
                        f"loss_last={dashboard.data_stats.loss_last:.4f} "
                        f"eta={dashboard.data_stats.eta_s:.1f}s"
                    )
                dashboard.sample_gpu_stats()
                live.update(dashboard.render())
                time.sleep(1.0 / max(args.refresh_hz, 1.0))

            dashboard.set_control_state("collect", "Collecting final stats from actors")
            mesh_results = trainers.finalize.call().get()
            dashboard.finalize_data(mesh_results, batch_size=args.batch_size, seq_len=args.seq_len)
            for point, result in mesh_results.items():
                dashboard.log(
                    f"[DATA] final point={point} rank={result['rank']} gpu={result['device']} "
                    f"loss_start={result['start_loss']:.4f} loss_end={result['end_loss']:.4f} "
                    f"avg={result['avg_loss']:.4f} best={result['best_loss']:.4f} "
                    f"tok/s={result['tokens_per_sec']:.1f} step_ms={result['step_time_ms']:.1f} "
                    f"peak_mem_mb={result['peak_mem_mb']:.1f}"
                )
            dashboard.sample_gpu_stats()
            live.update(dashboard.render())
        finally:
            dashboard.set_control_state("teardown", "Stopping ProcMesh")
            procs.stop().get()
            dashboard.sample_gpu_stats()
            dashboard.set_control_state("done", "ProcMesh stopped")
            live.update(dashboard.render())

    if mesh_results is None:
        raise RuntimeError("Training did not produce results")

    console.print("\nMonarch tiny-transformer training complete")
    for point, result in mesh_results.items():
        console.print(
            f"{point}: rank={result['rank']} gpu={result['device']} "
            f"loss {result['start_loss']:.4f}->{result['end_loss']:.4f} "
            f"avg={result['avg_loss']:.4f} best={result['best_loss']:.4f} "
            f"tok/s={result['tokens_per_sec']:.1f} peak_mem_mb={result['peak_mem_mb']:.1f}"
        )


if __name__ == "__main__":
    main()
