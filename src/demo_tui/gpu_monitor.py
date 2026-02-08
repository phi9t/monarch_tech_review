"""GPU statistics sampling via nvidia-smi with torch.cuda fallback."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class GpuStats:
    """Snapshot of a single GPU's utilization and memory."""

    index: int
    name: str
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float


class GpuMonitor:
    """Sample GPU stats from nvidia-smi, falling back to torch.cuda."""

    def __init__(self) -> None:
        self._last_sample_time: float = 0.0
        self._last_error: str = ""

    @property
    def last_sample_time(self) -> float:
        return self._last_sample_time

    @property
    def last_error(self) -> str:
        return self._last_error

    def sample(self) -> list[GpuStats]:
        """Query GPU stats. Tries nvidia-smi first, then torch.cuda fallback."""
        self._last_sample_time = time.perf_counter()
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

            stats: list[GpuStats] = []
            for line in lines:
                idx, name, util, mem_used, mem_total, temp = [
                    p.strip() for p in line.split(",")
                ]
                stats.append(
                    GpuStats(
                        index=int(idx),
                        name=name,
                        utilization_pct=float(util),
                        memory_used_mb=float(mem_used),
                        memory_total_mb=float(mem_total),
                        temperature_c=float(temp),
                    )
                )
            self._last_error = ""
            return stats

        except Exception as exc:
            self._last_error = str(exc)
            return self._torch_fallback()

    def _torch_fallback(self) -> list[GpuStats]:
        """Fall back to torch.cuda.mem_get_info when nvidia-smi is unavailable."""
        stats: list[GpuStats] = []
        try:
            import torch  # noqa: F811
        except ImportError:
            return stats

        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                used = total - free
                stats.append(
                    GpuStats(
                        index=i,
                        name="(torch fallback)",
                        utilization_pct=0.0,
                        memory_used_mb=used / (1024**2),
                        memory_total_mb=total / (1024**2),
                        temperature_c=0.0,
                    )
                )
            except Exception:
                stats.append(
                    GpuStats(
                        index=i,
                        name="(unavailable)",
                        utilization_pct=0.0,
                        memory_used_mb=0.0,
                        memory_total_mb=0.0,
                        temperature_c=0.0,
                    )
                )
        return stats

    @staticmethod
    def format_gpu_line(stat: GpuStats) -> str:
        """Format a GpuStats into a single display line."""
        mem_pct = (stat.memory_used_mb / max(stat.memory_total_mb, 1.0)) * 100.0
        bar_len = max(1, min(10, int(round(stat.utilization_pct / 10.0))))
        util_bar = "\u2588" * bar_len
        return (
            f"GPU{stat.index} {stat.name[:16]:<16} "
            f"util {stat.utilization_pct:>5.1f}% {util_bar:<10} "
            f"mem {stat.memory_used_mb:>5.0f}/{stat.memory_total_mb:.0f}MB "
            f"({mem_pct:>5.1f}%) "
            f"temp {stat.temperature_c:>4.0f}C"
        )
