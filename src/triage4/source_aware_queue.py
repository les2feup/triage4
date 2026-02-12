"""
Two-level round-robin queue for alarm sources.

Provides fairness across alarm zones (sources) and devices within each zone.
"""

from collections import deque
from typing import Deque, Dict, Optional


class SourceAwareQueue:
    """Round-robin across zones, then devices within each zone."""

    def __init__(self):
        self.zone_queues: Dict[int, Dict[str, Deque[int]]] = {}
        self.last_served_zone: Optional[int] = None
        self.last_served_device: Dict[int, Optional[str]] = {}

    def enqueue(self, job_idx: int, device_id: str, zone_id: int) -> None:
        """Add job to the queue for its zone/device."""
        if zone_id not in self.zone_queues:
            self.zone_queues[zone_id] = {}
            self.last_served_device[zone_id] = None

        if device_id not in self.zone_queues[zone_id]:
            self.zone_queues[zone_id][device_id] = deque()

        self.zone_queues[zone_id][device_id].append(job_idx)

    def dequeue(self) -> Optional[int]:
        """Pop next job using zone-level then device-level round robin."""
        if not self.zone_queues:
            return None

        zones = sorted(self.zone_queues.keys())
        start_zone_idx = 0
        if self.last_served_zone is not None:
            for i, z in enumerate(zones):
                if z > self.last_served_zone:
                    start_zone_idx = i
                    break

        for i in range(len(zones)):
            zone_idx = (start_zone_idx + i) % len(zones)
            zone_id = zones[zone_idx]
            job_idx = self._dequeue_from_zone(zone_id)
            if job_idx is not None:
                self.last_served_zone = zone_id
                return job_idx

        return None

    def _dequeue_from_zone(self, zone_id: int) -> Optional[int]:
        """Dequeue respecting per-device fairness inside a zone."""
        devices = sorted(self.zone_queues[zone_id].keys())
        if not devices:
            del self.zone_queues[zone_id]
            self.last_served_device.pop(zone_id, None)
            return None

        start_dev_idx = 0
        last_dev = self.last_served_device.get(zone_id)
        if last_dev is not None:
            for i, dev in enumerate(devices):
                if dev > last_dev:
                    start_dev_idx = i
                    break

        for i in range(len(devices)):
            idx = (start_dev_idx + i) % len(devices)
            device_id = devices[idx]
            queue = self.zone_queues[zone_id][device_id]
            if queue:
                job_idx = queue.popleft()
                self.last_served_device[zone_id] = device_id
                if not queue:
                    del self.zone_queues[zone_id][device_id]
                if not self.zone_queues[zone_id]:
                    del self.zone_queues[zone_id]
                    self.last_served_device.pop(zone_id, None)
                return job_idx

        return None

    def is_empty(self) -> bool:
        """True if no jobs queued."""
        return len(self.zone_queues) == 0

    def queue_length(self) -> int:
        """Total queued jobs."""
        total = 0
        for device_map in self.zone_queues.values():
            total += sum(len(q) for q in device_map.values())
        return total

    def zone_count(self) -> int:
        """Number of active zones."""
        return len(self.zone_queues)

    def __repr__(self) -> str:
        return (
            f"SourceAwareQueue(zones={self.zone_count()}, "
            f"jobs={self.queue_length()}, last_zone={self.last_served_zone})"
        )

