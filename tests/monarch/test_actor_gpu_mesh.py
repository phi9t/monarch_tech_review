import pytest


@pytest.mark.gpu
def test_actor_mesh_runs_on_real_gpus():
    import torch
    from monarch.actor import Actor, current_rank, endpoint, this_host

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        pytest.skip(f"Need >=2 GPUs for mesh test, found {gpu_count}")

    class GpuProbeActor(Actor):
        @endpoint
        def probe(self) -> dict:
            rank = current_rank().rank
            device = rank % torch.cuda.device_count()
            torch.cuda.set_device(device)
            x = torch.randn(8, device="cuda")
            return {
                "rank": rank,
                "device": torch.cuda.current_device(),
                "sum": float(x.sum().item()),
            }

    host = this_host()
    procs = host.spawn_procs(per_host={"gpus": 2})
    actors = procs.spawn("gpu_probe", GpuProbeActor)
    results = actors.probe.call().get()
    procs.stop().get()

    values = list(results.values())
    assert len(values) == 2
    assert {item["rank"] for item in values} == {0, 1}
    assert {item["device"] for item in values} == {0, 1}
