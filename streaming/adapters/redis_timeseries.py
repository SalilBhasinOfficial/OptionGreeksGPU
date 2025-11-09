"""Redis TimeSeries adapter for time-series data with aggregations."""

import time
from typing import List, Generator, Optional, Tuple
from datetime import datetime

from .base import BaseAdapter
from ..models import MarketUpdate, OptionType, DataSource


class RedisTimeSeriesAdapter(BaseAdapter):
    """Adapter for Redis TimeSeries module."""

    def __init__(self, redis_client, key_prefix: str = "ts:market"):
        super().__init__("RedisTimeSeriesAdapter")
        self.redis = redis_client
        self.key_prefix = key_prefix

    def connect(self) -> bool:
        try:
            self.redis.ping()
            self.is_connected = True
            return True
        except Exception:
            return False

    def disconnect(self):
        self.is_connected = False

    def read_batch(self, limit: Optional[int] = None) -> List[MarketUpdate]:
        # TimeSeries is primarily for writing, not batch reading
        return []

    def stream(self) -> Generator[List[MarketUpdate], None, None]:
        # Not typically used for streaming reads
        while True:
            yield []
            time.sleep(1)

    def get_range(self, contract_id: str, metric: str,
                  start_ts: int, end_ts: int) -> List[Tuple[int, float]]:
        """Get time-series data for a metric."""
        key = f"{self.key_prefix}:{contract_id}:{metric}"
        try:
            result = self.redis.execute_command('TS.RANGE', key, start_ts, end_ts)
            return [(ts, float(val)) for ts, val in result]
        except:
            return []

    def add_sample(self, contract_id: str, metric: str, timestamp: int, value: float):
        """Add a sample to time-series."""
        key = f"{self.key_prefix}:{contract_id}:{metric}"
        try:
            self.redis.execute_command('TS.ADD', key, timestamp, value)
        except:
            pass
