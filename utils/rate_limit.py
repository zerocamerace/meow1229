from __future__ import annotations

import time
from collections import deque
from typing import Deque

_RATE_LIMIT_BUCKETS: dict[str, Deque[float]] = {}


def is_rate_limited(key: str, max_events: int, window_seconds: int) -> bool:
    now = time.time()
    bucket = _RATE_LIMIT_BUCKETS.setdefault(key, deque())
    cutoff = now - window_seconds
    while bucket and bucket[0] <= cutoff:
        bucket.popleft()
    if len(bucket) >= max_events:
        return True
    bucket.append(now)
    return False
