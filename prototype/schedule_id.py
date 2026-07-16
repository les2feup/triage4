"""
Fingerprint a replay schedule so results can be tied to the workload that produced them.

A ``msg_id`` is assigned by arrival index, so it identifies a *position* in the
schedule, not a message. Regenerating a workload with different jitter keeps every
id and reshuffles what each one refers to: results collected under the old schedule
still join cleanly against the new one and silently describe different messages.
That failure is invisible — the joins succeed, the counts match, the numbers are
merely wrong.

The fingerprint covers the fields a join depends on (id, device, zone, alarm flag,
arrival time). Clients stamp it into their RTT shards; consolidation recomputes it
from the schedule on disk and refuses to proceed when the two disagree.

Stdlib only: this is imported both by the isolated prototype venv and by the main
repository venv.
"""

import hashlib
import json
from typing import List

# Short enough to sit in a CSV column and read in a diff, long enough that a
# collision between two schedules of one project is not a practical concern.
_DIGEST_CHARS = 12


def fingerprint(messages: List[dict]) -> str:
    """Return a stable short digest of a schedule's messages.

    Canonicalised field-by-field rather than hashing the raw file, so formatting
    (indentation, key order, float repr) cannot change the identity of a schedule
    that is otherwise the same. Arrival times are rounded to nanoseconds, which is
    finer than anything the replay can honour.
    """
    canonical = [
        [
            m["msg_id"],
            m["device_id"],
            int(m["zone_priority"]),
            bool(m["is_alarm"]),
            round(float(m["t"]), 9),
        ]
        for m in messages
    ]
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:_DIGEST_CHARS]


def fingerprint_file(path: str) -> str:
    """Fingerprint the schedule stored at ``path``."""
    with open(path) as handle:
        return fingerprint(json.load(handle)["messages"])
