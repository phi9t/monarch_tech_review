import json
import textwrap

import pytest


@pytest.mark.gpu
@pytest.mark.spmd
def test_spmd_serve_torchrun_gpu_smoke(tmp_path):
    import torch
    from monarch.job.spmd import serve

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        pytest.skip(f"Need >=2 GPUs for SPMD smoke, found {gpu_count}")

    out_dir = tmp_path / "spmd_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_path / "spmd_smoke.py"
    script_path.write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path

            import torch

            rank = int(os.environ.get("RANK", -1))
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            world = int(os.environ.get("WORLD_SIZE", -1))
            torch.cuda.set_device(local_rank)
            x = torch.randn(16, device="cuda")
            payload = {{
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world,
                "device": torch.cuda.current_device(),
                "sum": float(x.sum().item()),
            }}
            out = Path("{out_dir}") / f"rank_{{rank}}.json"
            out.write_text(json.dumps(payload))
            print(json.dumps(payload))
            """
        ),
        encoding="utf-8",
    )

    job = serve(
        [
            "torchrun",
            "--nproc-per-node=2",
            "--standalone",
            str(script_path),
        ],
        scheduler="local_cwd",
    )
    job.run_spmd()

    outputs = sorted(out_dir.glob("rank_*.json"))
    assert len(outputs) == 2

    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in outputs]
    assert {p["rank"] for p in payloads} == {0, 1}
    assert all(p["world_size"] == 2 for p in payloads)
    assert {p["device"] for p in payloads} == {0, 1}
