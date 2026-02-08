import uuid

import pytest


@pytest.mark.gpu
def test_service_registry_and_health_on_gpu_workers():
    import torch
    from monarch.actor import Actor, current_rank, endpoint, this_host
    from monarch_utils.services import (
        Service,
        get_service,
        list_services,
        register_service,
    )

    if torch.cuda.device_count() < 1:
        pytest.skip("Need at least one CUDA GPU")

    class Worker(Actor):
        @endpoint
        def ping(self) -> bool:
            return True

        @endpoint
        def work(self) -> dict:
            rank = current_rank().rank
            device = rank % torch.cuda.device_count()
            torch.cuda.set_device(device)
            x = torch.randn(4, device="cuda")
            return {
                "rank": rank,
                "device": torch.cuda.current_device(),
                "sum": float(x.sum().item()),
            }

    host = this_host()
    procs = host.spawn_procs(per_host={"procs": 2})
    svc_mesh = procs.slice(procs=0)
    worker_procs = procs.slice(procs=slice(0, 2))

    service_name = f"svc_{uuid.uuid4().hex[:8]}"
    service = svc_mesh.spawn(
        f"service_{service_name}",
        Service,
        service_name=service_name,
        worker_class=Worker,
        procs=worker_procs,
        procs_per_replica=1,
    )

    register_service(service_name, service)
    assert service_name in list_services()

    discovered = get_service(service_name)
    replica, idx = discovered.get_replica_with_idx.call_one().get()
    payload = replica.work.call_one().get()
    assert payload["device"] in {0, 1}
    assert isinstance(payload["sum"], float)

    discovered.mark_unhealthy.call_one(idx).get()
    status = discovered.get_health_status.call_one().get()
    assert status["unhealthy"] == 1

    recovered = discovered.check_health.call_one().get()
    assert idx in recovered["recovered"]
    status2 = discovered.get_health_status.call_one().get()
    assert status2["healthy"] == 2
    assert status2["unhealthy"] == 0

    procs.stop().get()
