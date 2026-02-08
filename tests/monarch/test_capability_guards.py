import pytest


# References investigated for this repo's Monarch API surface:
# - https://meta-pytorch.org/monarch/
# - https://meta-pytorch.org/monarch/generated/examples/index.html
# - https://docs.pytorch.org/tutorials/intermediate/monarch_distributed_tutorial.html
# - https://meta-pytorch.org/monarch/generated/examples/ping_pong.html
# - https://meta-pytorch.org/monarch/actors.html#error-handling-in-meshes


def test_notebook_referenced_monarch_symbols_exist():
    from monarch.actor import Actor, current_rank, endpoint, this_host
    from monarch.job.spmd import serve
    from monarch.spmd import SPMDActor

    assert Actor is not None
    assert endpoint is not None
    assert current_rank is not None
    assert this_host is not None
    assert SPMDActor is not None
    assert serve is not None


@pytest.mark.rdma_optional
def test_rdma_capability_smoke_or_skip():
    from monarch.rdma import is_rdma_available

    if not is_rdma_available():
        pytest.skip("RDMA is not available in this Zephyr environment")

    from monarch.rdma import RDMABuffer

    assert RDMABuffer is not None
