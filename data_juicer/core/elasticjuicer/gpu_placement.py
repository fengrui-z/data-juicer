"""Concrete GPU placement using one whole-device Ray reservation per bundle.

Bundle indices are logical names. A GPU task first measures their node/UUID
mapping; workers then use explicit bundle indices and verify that mapping.
This reserves scheduling resources, not a hardware VRAM isolation boundary.
"""

import copy
import os


def inspect_gpu_bundle():
    import ray
    import torch

    context = ray.get_runtime_context()
    gpu_ids = context.get_accelerator_ids().get("GPU", [])
    if len(gpu_ids) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("a GPU placement bundle must own exactly one visible device")
    device = torch.cuda.get_device_properties(0)
    return {
        "node_id": context.get_node_id(),
        "gpu_uuid": str(device.uuid),
        "ray_gpu_id": str(gpu_ids[0]),
        "memory_bytes": device.total_memory,
        "gpu_name": device.name,
    }


class GpuPlacementLease:
    def __init__(self, num_devices, cpus_per_device, timeout_seconds=60):
        import ray
        from ray.util.placement_group import placement_group, remove_placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if num_devices < 1 or cpus_per_device <= 0:
            raise ValueError("GPU placement requires positive devices and CPU resources")
        self.group = placement_group([{"GPU": 1, "CPU": cpus_per_device} for _ in range(num_devices)], strategy="PACK")
        try:
            ray.get(self.group.ready(), timeout=timeout_seconds)
            discover = ray.remote(num_cpus=0, num_gpus=1, max_calls=1)(inspect_gpu_bundle)
            references = [
                discover.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.group,
                        placement_group_bundle_index=index,
                    )
                ).remote()
                for index in range(num_devices)
            ]
            self.topology = [
                {"device_id": index, **identity}
                for index, identity in enumerate(ray.get(references, timeout=timeout_seconds))
            ]
            if len({(device["node_id"], device["gpu_uuid"]) for device in self.topology}) != num_devices:
                raise RuntimeError("GPU bundles do not map one-to-one to physical devices")
        except BaseException:
            remove_placement_group(self.group)
            raise

    def close(self):
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(self.group)


class PlacementSelector:
    """Ray Data per-actor scheduling arguments for one fixed-size pool."""

    def __init__(self, group, device_ids, topology, runtime_env=None):
        if not device_ids:
            raise ValueError("placement requires a nonempty actor device list")
        self.group = group
        self.device_ids = list(device_ids)
        self.topology = {item["device_id"]: item for item in topology}
        if any(device not in self.topology for device in device_ids):
            raise ValueError("planned device is absent from the physical topology")
        self.runtime_env = dict(runtime_env or {})
        self.next_slot = 0

    def __call__(self):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        # Ray also evaluates this callback for diagnostic resource queries.
        # Cycling is harmless there; the execution sink, not callback count,
        # enforces one acknowledged actor per slot.
        slot = self.next_slot % len(self.device_ids)
        self.next_slot += 1
        device = self.topology[self.device_ids[slot]]
        runtime_env = copy.deepcopy(self.runtime_env)
        runtime_env.setdefault("env_vars", {}).update(
            {
                "ELASTIC_JUICER_PLACEMENT_SLOT": str(slot),
                "ELASTIC_JUICER_GPU_UUID": device["gpu_uuid"],
                "ELASTIC_JUICER_NODE_ID": device["node_id"],
            }
        )
        return {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=self.group,
                placement_group_bundle_index=device["device_id"],
                placement_group_capture_child_tasks=False,
            ),
            "runtime_env": runtime_env,
        }


def verify_worker_placement(identity, placement):
    slot = int(os.environ.get("ELASTIC_JUICER_PLACEMENT_SLOT", "-1"))
    if not 0 <= slot < len(placement["device_ids"]):
        raise RuntimeError("actor has no valid physical placement slot")
    device_id = placement["device_ids"][slot]
    device = next(item for item in placement["topology"] if item["device_id"] == device_id)
    if identity["node_id"] != device["node_id"] or identity["gpu_uuids"] != [device["gpu_uuid"]]:
        raise RuntimeError("actual node/GPU UUID does not match the placement envelope")
    return {"placement_slot": slot, "device_id": device_id, "placement_verified": True}
