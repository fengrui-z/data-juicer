"""Top-level coordinator for the Tower/Captain control loop."""

import threading
from typing import Dict, Optional

import psutil

from .contracts import AllocationPlan, ClusterState, TowerMode
from .scheduler.captain import Captain, CaptainConfig
from .scheduler.tower import Tower


def detect_cluster_state() -> ClusterState:
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count(logical=True) or 1
    return ClusterState(
        total_cpu_cores=cpu_count,
        total_memory_mb=memory.total / 1024**2,
        total_gpu_count=0,
        available_cpu_cores=float(cpu_count),
        available_memory_mb=memory.available / 1024**2,
        available_gpus=0,
    )


class ElasticJuicer:
    """Own a Tower and its registered Captains with explicit lifecycle."""

    def __init__(
        self,
        cluster_state: Optional[ClusterState] = None,
        tower_mode: TowerMode | str = TowerMode.SHADOW,
        rebalance_interval_sec: float = 5.0,
        **tower_kwargs,
    ):
        if rebalance_interval_sec <= 0:
            raise ValueError("rebalance_interval_sec must be positive")
        self.rebalance_interval_sec = rebalance_interval_sec
        self.tower = Tower(
            cluster_state=cluster_state or detect_cluster_state(),
            mode=tower_mode,
            update_interval_sec=rebalance_interval_sec,
            **tower_kwargs,
        )
        self.captains: Dict[str, Captain] = {}
        self._captain_ids: Dict[str, str] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def register_captain(self, config: CaptainConfig) -> Captain:
        with self._lock:
            existing = self.captains.get(config.stage_name)
            if existing is not None:
                return existing
            captain_id = self.tower.register_stage(config.stage_name)
            captain = Captain(config)
            self.tower.register_controller(captain_id, captain)
            self.captains[config.stage_name] = captain
            self._captain_ids[config.stage_name] = captain_id
            return captain

    def tick(self, force: bool = True) -> AllocationPlan:
        with self._lock:
            return self.tower.rebalance(force=force)

    def start(self):
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ElasticJuicer-Tower",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.rebalance_interval_sec * 2)
            self._thread = None

    def _run_loop(self):
        while not self._stop_event.is_set():
            self.tick(force=True)
            self._stop_event.wait(self.rebalance_interval_sec)

    def get_status(self) -> dict:
        return {
            "is_running": self.is_running,
            "tower": self.tower.get_global_stats(),
            "captains": {
                stage_name: captain.get_stats()
                for stage_name, captain in self.captains.items()
            },
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
