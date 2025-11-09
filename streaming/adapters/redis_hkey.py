"""
Redis HKEY adapter for polling current market snapshots.

Use case: Reading current prices from Redis hashes updated by external systems.
"""

import time
from typing import List, Generator, Optional, Dict, Set
from datetime import datetime

from .base import BaseAdapter
from ..models import MarketUpdate, OptionType, DataSource


class RedisHKEYAdapter(BaseAdapter):
    """
    Adapter for polling Redis hash keys for current market data.
    """

    def __init__(self, redis_client, key_pattern: str = "nse:options:*",
                 poll_interval_ms: int = 100):
        """
        Initialize Redis HKEY adapter.

        Args:
            redis_client: Redis client instance
            key_pattern: Pattern for keys to scan (e.g., "nse:options:*")
            poll_interval_ms: Polling interval in milliseconds
        """
        super().__init__("RedisHKEYAdapter")
        self.redis = redis_client
        self.key_pattern = key_pattern
        self.poll_interval_ms = poll_interval_ms

        # Track last seen values for change detection
        self.last_values: Dict[str, Dict] = {}

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis.ping()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            return False

    def disconnect(self):
        """Disconnect."""
        self.is_connected = False

    def read_batch(self, limit: Optional[int] = None) -> List[MarketUpdate]:
        """
        Read current snapshot from all matching Redis hashes.

        Args:
            limit: Maximum number of updates to read

        Returns:
            List of MarketUpdate objects
        """
        if not self.is_connected:
            self.connect()

        try:
            # Scan for matching keys
            keys = list(self.redis.scan_iter(match=self.key_pattern, count=1000))

            if limit:
                keys = keys[:limit]

            updates = []
            for key in keys:
                # Get hash fields
                data = self.redis.hgetall(key)
                if data:
                    update = self._parse_hash(key, data)
                    if update and self.validate_update(update):
                        updates.append(update)

            return updates

        except Exception as e:
            print(f"Error reading from Redis: {e}")
            return []

    def stream(self) -> Generator[List[MarketUpdate], None, None]:
        """
        Poll Redis hashes continuously and yield updates when values change.

        Yields:
            Batches of MarketUpdate objects (only changed values)
        """
        if not self.is_connected:
            self.connect()

        while True:
            try:
                # Scan for matching keys
                keys = list(self.redis.scan_iter(match=self.key_pattern, count=1000))

                updates = []
                for key in keys:
                    # Get hash fields
                    data = self.redis.hgetall(key)

                    if data and self._has_changed(key, data):
                        update = self._parse_hash(key, data)
                        if update and self.validate_update(update):
                            updates.append(update)

                        # Update last seen value
                        self.last_values[key] = data

                if updates:
                    yield updates

                # Sleep before next poll
                time.sleep(self.poll_interval_ms / 1000.0)

            except Exception as e:
                print(f"Error in stream processing: {e}")
                time.sleep(1)

    def _parse_hash(self, key: str, data: Dict[bytes, bytes]) -> Optional[MarketUpdate]:
        """
        Parse Redis hash into MarketUpdate.

        Key format: "nse:options:NIFTY:18000:2025-01-30:CE"
        Hash fields: underlying_price, call_price, put_price, timestamp, etc.

        Args:
            key: Redis key (bytes or str)
            data: Hash fields

        Returns:
            MarketUpdate or None if parsing fails
        """
        try:
            # Parse key
            if isinstance(key, bytes):
                key = key.decode()

            parts = key.split(':')
            if len(parts) < 6:
                return None

            symbol = parts[2]
            strike = float(parts[3])
            expiry = datetime.fromisoformat(parts[4])
            option_type_str = parts[5].upper()

            if option_type_str in ['CE', 'CALL']:
                option_type = OptionType.CALL
            elif option_type_str in ['PE', 'PUT']:
                option_type = OptionType.PUT
            else:
                return None

            contract_id = f"{symbol}:{strike}:{expiry.strftime('%Y-%m-%d')}:{option_type.value}"

            # Decode hash fields
            decoded = {k.decode() if isinstance(k, bytes) else k:
                      v.decode() if isinstance(v, bytes) else v
                      for k, v in data.items()}

            # Create MarketUpdate
            update = MarketUpdate(
                contract_id=contract_id,
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                underlying_price=float(decoded.get('underlying_price', 0)),
                call_price=float(decoded.get('call_price', 0)) or None,
                put_price=float(decoded.get('put_price', 0)) or None,
                timestamp=float(decoded.get('timestamp', time.time())),
                source=DataSource.REDIS_HKEY,
                volume=int(decoded['volume']) if 'volume' in decoded else None,
                open_interest=int(decoded['open_interest']) if 'open_interest' in decoded else None
            )

            return update

        except Exception as e:
            print(f"Error parsing hash: {e}")
            return None

    def _has_changed(self, key: str, data: Dict) -> bool:
        """
        Check if hash has changed since last read.

        Args:
            key: Redis key
            data: Current hash data

        Returns:
            True if changed or first time seeing this key
        """
        if key not in self.last_values:
            return True

        # Compare timestamp if available
        current_ts = data.get(b'timestamp', data.get('timestamp'))
        last_ts = self.last_values[key].get(b'timestamp', self.last_values[key].get('timestamp'))

        if current_ts and last_ts:
            return current_ts != last_ts

        # Fallback: compare all fields
        return data != self.last_values[key]
