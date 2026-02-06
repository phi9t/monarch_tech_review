"""
Service infrastructure for Monarch actors.

Provides:
- ServiceRegistry: Singleton for service discovery
- Service: Manages worker replicas with health tracking and routing
- get_service/register_service: Helper functions for discovery
"""

from monarch.actor import Actor, endpoint, get_or_spawn_controller


class ServiceRegistry(Actor):
    """
    Singleton registry for service discovery.

    Found via get_or_spawn_controller("service_registry", ServiceRegistry).
    First caller spawns it, subsequent callers get the existing one.
    """

    def __init__(self):
        self.services: dict[str, Actor] = {}
        print("[REGISTRY] ServiceRegistry spawned")

    @endpoint
    def register(self, name: str, service) -> None:
        """Register a service by name."""
        self.services[name] = service
        print(f"[REGISTRY] Registered '{name}'")

    @endpoint
    def get(self, name: str):
        """Get a service by name."""
        if name not in self.services:
            raise KeyError(f"Service '{name}' not found")
        return self.services[name]

    @endpoint
    def list_services(self) -> list[str]:
        """List all registered service names."""
        return list(self.services.keys())


def _get_registry():
    """Get the singleton ServiceRegistry."""
    return get_or_spawn_controller("service_registry", ServiceRegistry).get()


def get_service(name: str):
    """Get a service by name. Raises KeyError if not found."""
    registry = _get_registry()
    return registry.get.call_one(name).get()


def register_service(name: str, service) -> None:
    """Register a service so others can find it by name."""
    registry = _get_registry()
    registry.register.call_one(name, service).get()


def list_services() -> list[str]:
    """List all registered service names."""
    registry = _get_registry()
    return registry.list_services.call_one().get()


class Service(Actor):
    """
    Manages a pool of worker replicas with routing and health tracking.

    Each replica is an ActorMesh (potentially spanning multiple GPUs/hosts).
    The service:
    1. Routes requests round-robin to healthy replicas
    2. Tracks health and handles failures
    3. Can probe unhealthy replicas to check if they've recovered

    Usage:
        # Spawn Service actor
        svc = procs.spawn("my_service", Service,
                          service_name="generators", num_replicas=4)

        # Set worker replicas (spawned separately)
        svc.set_replicas.call_one(workers).get()

        # Get a replica (round-robin)
        worker, idx = svc.get_replica_with_idx.call_one().get()

        # On failure, mark unhealthy
        svc.mark_unhealthy.call_one(idx).get()
    """

    def __init__(self, service_name: str, num_replicas: int):
        self.service_name = service_name
        self.num_replicas = num_replicas
        self.replicas = None  # Set via set_replicas
        self.healthy = set(range(num_replicas))
        self.unhealthy: set[int] = set()
        self.next_idx = 0
        print(f"[SERVICE:{service_name}] Initialized for {num_replicas} replicas")

    @endpoint
    def set_replicas(self, replicas) -> None:
        """Set the replica actors."""
        self.replicas = replicas
        print(f"[SERVICE:{self.service_name}] {len(replicas)} replicas connected")

    @endpoint
    def ping(self) -> bool:
        """Health check."""
        return True

    @endpoint
    def get_replica(self):
        """Get a healthy replica (round-robin selection)."""
        if not self.healthy:
            raise RuntimeError("No healthy replicas available!")
        healthy_list = sorted(self.healthy)
        idx = self.next_idx % len(healthy_list)
        self.next_idx += 1
        return self.replicas[healthy_list[idx]]

    @endpoint
    def get_replica_with_idx(self):
        """Get a healthy replica and its index (for marking unhealthy later)."""
        if not self.healthy:
            raise RuntimeError("No healthy replicas available!")
        healthy_list = sorted(self.healthy)
        idx = self.next_idx % len(healthy_list)
        self.next_idx += 1
        replica_idx = healthy_list[idx]
        return self.replicas[replica_idx], replica_idx

    @endpoint
    def mark_unhealthy(self, replica_idx: int) -> None:
        """Mark a replica as unhealthy."""
        if replica_idx in self.healthy:
            self.healthy.discard(replica_idx)
            self.unhealthy.add(replica_idx)
            print(f"[SERVICE:{self.service_name}] Replica {replica_idx} marked unhealthy. "
                  f"Healthy: {len(self.healthy)}/{self.num_replicas}")

    @endpoint
    def mark_healthy(self, replica_idx: int) -> None:
        """Mark a replica as healthy again."""
        if replica_idx < self.num_replicas:
            self.unhealthy.discard(replica_idx)
            self.healthy.add(replica_idx)

    @endpoint
    def check_health(self) -> dict:
        """
        Probe all unhealthy replicas to see if they've recovered.
        Returns a dict with recovery results.
        """
        recovered = []
        still_unhealthy = []

        for replica_idx in list(self.unhealthy):
            try:
                self.replicas[replica_idx].ping.call_one().get()
                self.unhealthy.discard(replica_idx)
                self.healthy.add(replica_idx)
                recovered.append(replica_idx)
                print(f"[SERVICE:{self.service_name}] Replica {replica_idx} recovered!")
            except Exception:
                still_unhealthy.append(replica_idx)

        return {
            "recovered": recovered,
            "still_unhealthy": still_unhealthy,
            "healthy_count": len(self.healthy),
        }

    @endpoint
    def get_health_status(self) -> dict:
        """Get current health status."""
        return {
            "total": self.num_replicas,
            "healthy": len(self.healthy),
            "unhealthy": len(self.unhealthy),
            "healthy_indices": sorted(self.healthy),
            "unhealthy_indices": sorted(self.unhealthy),
        }
