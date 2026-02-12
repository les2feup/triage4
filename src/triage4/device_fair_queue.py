"""
Per-device fair queue with round-robin device selection.

Prevents device monopolization by cycling through devices and serving
one message per device per round, maintaining FIFO within each device.
"""

from collections import deque
from typing import Dict, Optional


class DeviceFairQueue:
    """
    Round-robin per-device queue within a band.

    Ensures fairness by cycling through active devices and serving one message
    from each device per round. Messages within each device are served FIFO.

    Key properties:
        - Fair device selection: Each device gets equal turns regardless of load
        - FIFO within device: Messages from same device served in arrival order
        - Monopolization prevention: High-rate device cannot starve low-rate device
        - Dynamic device set: Devices added/removed as queues fill/empty

    Example:
        >>> queue = DeviceFairQueue()
        >>> queue.enqueue(job_idx=0, device_id="A")
        >>> queue.enqueue(job_idx=1, device_id="B")
        >>> queue.enqueue(job_idx=2, device_id="A")
        >>> queue.dequeue()  # Returns 0 from device A
        0
        >>> queue.dequeue()  # Returns 1 from device B (round-robin)
        1
        >>> queue.dequeue()  # Returns 2 from device A (back to A)
        2
    """

    def __init__(self):
        """Initialize empty per-device fair queue."""
        self.device_queues: Dict[str, deque[int]] = {}
        self.last_served_device: Optional[str] = None

    def enqueue(self, job_idx: int, device_id: str) -> None:
        """
        Add message to device's queue.

        Args:
            job_idx: Index of job in arrival_times array
            device_id: Device identifier (e.g., "sensor_42", "EDU_5")
        """
        if device_id not in self.device_queues:
            self.device_queues[device_id] = deque()
        self.device_queues[device_id].append(job_idx)

    def dequeue(self) -> Optional[int]:
        """
        Remove and return next message using round-robin device selection.

        Round-robin algorithm:
            1. Get list of devices with messages
            2. Find position of last_served_device (or start at 0)
            3. Iterate circularly starting from next device
            4. Return first message from first non-empty device
            5. Update last_served_device pointer
            6. Clean up empty device queues

        Returns:
            Job index of next message, or None if all queues empty

        Example:
            With devices ["A", "B", "C"] and last_served="A":
            - Next check: B (if non-empty, serve from B)
            - Then: C, then A, then B, ... (circular)
        """
        if not self.device_queues:
            return None

        # Get sorted device list for deterministic round-robin
        devices = sorted(self.device_queues.keys())

        # Find starting position (next device after last served)
        # Maintain circular order even when devices are removed
        start_idx = 0
        if self.last_served_device is not None:
            # Find first device that comes after last_served in sorted order
            for i, dev in enumerate(devices):
                if dev > self.last_served_device:
                    start_idx = i
                    break
            # If no device comes after (last_served was alphabetically last),
            # wrap around to start

        # Search circularly for next non-empty device
        for i in range(len(devices)):
            idx = (start_idx + i) % len(devices)
            device_id = devices[idx]

            if self.device_queues[device_id]:
                # Serve from this device
                self.last_served_device = device_id
                job_idx = self.device_queues[device_id].popleft()

                # Clean up empty queues
                if not self.device_queues[device_id]:
                    del self.device_queues[device_id]

                return job_idx

        # All queues empty (shouldn't reach here if self.device_queues non-empty)
        return None

    def peek(self) -> Optional[int]:
        """
        View next message without removing it.

        Uses same round-robin logic as dequeue() but doesn't modify state.

        Returns:
            Job index of next message, or None if all queues empty
        """
        if not self.device_queues:
            return None

        devices = sorted(self.device_queues.keys())
        start_idx = 0
        if self.last_served_device is not None:
            # Find first device that comes after last_served in sorted order
            for i, dev in enumerate(devices):
                if dev > self.last_served_device:
                    start_idx = i
                    break

        for i in range(len(devices)):
            idx = (start_idx + i) % len(devices)
            device_id = devices[idx]

            if self.device_queues[device_id]:
                return self.device_queues[device_id][0]

        return None

    def is_empty(self) -> bool:
        """
        Check if all device queues are empty.

        Returns:
            True if no messages queued, False otherwise
        """
        return len(self.device_queues) == 0

    def queue_length(self) -> int:
        """
        Get total number of messages across all devices.

        Returns:
            Total queued messages
        """
        return sum(len(q) for q in self.device_queues.values())

    def device_count(self) -> int:
        """
        Get number of active devices (devices with queued messages).

        Returns:
            Number of devices with non-empty queues
        """
        return len(self.device_queues)

    def get_device_queue_lengths(self) -> Dict[str, int]:
        """
        Get per-device queue lengths.

        Returns:
            Dictionary mapping device_id to queue length
        """
        return {device_id: len(queue) for device_id, queue in self.device_queues.items()}

    def __repr__(self) -> str:
        total = self.queue_length()
        devices = self.device_count()
        return (
            f"DeviceFairQueue(total_messages={total}, "
            f"active_devices={devices}, last_served={self.last_served_device})"
        )
