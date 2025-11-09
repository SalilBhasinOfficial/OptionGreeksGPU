"""
State management for streaming option Greeks computation.

Implements two-tier storage:
- Tier 1: In-memory LRU cache for hot data (sub-microsecond access)
- Tier 2: Redis for persistence (sub-millisecond access)
"""

import json
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .models import ContractState, MarketUpdate, OptionType


class StateManager:
    """
    Manages state for option contracts with two-tier storage.

    Architecture:
    - LRU cache for frequently accessed contracts
    - Redis backing store for persistence
    - IV history in Redis sorted sets
    """

    def __init__(self, redis_client=None, cache_size: int = 10000,
                 iv_history_depth: int = 100, state_ttl_days: int = 7):
        """
        Initialize state manager.

        Args:
            redis_client: Redis client instance (optional for testing)
            cache_size: Maximum number of contracts in LRU cache
            iv_history_depth: Number of IV values to keep in history
            state_ttl_days: TTL for state in Redis (days)
        """
        self.redis = redis_client
        self.cache_size = cache_size
        self.iv_history_depth = iv_history_depth
        self.state_ttl_seconds = state_ttl_days * 86400

        # In-memory LRU cache
        self.cache: OrderedDict[str, ContractState] = OrderedDict()

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'redis_reads': 0,
            'redis_writes': 0,
            'iv_history_writes': 0
        }

    def get_contract_state(self, contract_id: str) -> Optional[ContractState]:
        """
        Get current state for a contract.

        Checks cache first, then Redis if cache miss.

        Args:
            contract_id: Unique contract identifier

        Returns:
            ContractState if found, None otherwise
        """
        # Check cache
        if contract_id in self.cache:
            self.stats['cache_hits'] += 1
            # Move to end (most recently used)
            self.cache.move_to_end(contract_id)
            return self.cache[contract_id]

        # Cache miss
        self.stats['cache_misses'] += 1

        # Try Redis if available
        if self.redis:
            state = self._load_from_redis(contract_id)
            if state:
                # Add to cache
                self._add_to_cache(contract_id, state)
                return state

        return None

    def update_contract_state(self, contract_id: str, state: ContractState):
        """
        Update contract state in cache and Redis.

        Args:
            contract_id: Unique contract identifier
            state: Updated contract state
        """
        # Update timestamp and counter
        state.last_update = time.time()
        state.update_count += 1

        # Update cache
        self._add_to_cache(contract_id, state)

        # Write to Redis if available
        if self.redis:
            self._save_to_redis(contract_id, state)

    def update_from_market_update(self, update: MarketUpdate,
                                   call_greeks: Optional[Dict[str, float]] = None,
                                   put_greeks: Optional[Dict[str, float]] = None):
        """
        Update or create contract state from market update and Greeks.

        Args:
            update: Market update with prices
            call_greeks: Computed call Greeks
            put_greeks: Computed put Greeks
        """
        # Get existing state or create new
        state = self.get_contract_state(update.contract_id)

        if state is None:
            # Create new state
            state = ContractState(
                contract_id=update.contract_id,
                symbol=update.symbol,
                strike=update.strike,
                expiry=update.expiry,
                option_type=update.option_type,
                underlying_price=update.underlying_price,
                call_price=update.call_price,
                put_price=update.put_price
            )
        else:
            # Update existing state
            state.underlying_price = update.underlying_price
            if update.call_price is not None:
                state.call_price = update.call_price
            if update.put_price is not None:
                state.put_price = update.put_price

        # Update Greeks if provided
        if call_greeks:
            state.call_iv = call_greeks.get('iv')
            state.call_delta = call_greeks.get('delta')
            state.call_gamma = call_greeks.get('gamma')
            state.call_vega = call_greeks.get('vega')
            state.call_theta = call_greeks.get('theta')
            state.call_rho = call_greeks.get('rho')
            state.call_delta2 = call_greeks.get('delta2')

        if put_greeks:
            state.put_iv = put_greeks.get('iv')
            state.put_delta = put_greeks.get('delta')
            state.put_gamma = put_greeks.get('gamma')
            state.put_vega = put_greeks.get('vega')
            state.put_theta = put_greeks.get('theta')
            state.put_rho = put_greeks.get('rho')
            state.put_delta2 = put_greeks.get('delta2')

        # Save state
        self.update_contract_state(update.contract_id, state)

    def get_iv_history(self, contract_id: str, option_type: str = 'call',
                       limit: int = 10) -> List[float]:
        """
        Get IV history for a contract.

        Args:
            contract_id: Unique contract identifier
            option_type: 'call' or 'put'
            limit: Number of recent IVs to return

        Returns:
            List of IVs (most recent first)
        """
        if not self.redis:
            return []

        key = f"iv_history:{contract_id}:{option_type}"

        # ZREVRANGE to get latest values (sorted by timestamp descending)
        try:
            results = self.redis.zrevrange(key, 0, limit - 1, withscores=False)

            ivs = []
            for item in results:
                # Format: "timestamp:iv_value"
                parts = item.decode() if isinstance(item, bytes) else item
                parts = str(parts).split(':')
                if len(parts) >= 2:
                    try:
                        ivs.append(float(parts[1]))
                    except ValueError:
                        continue

            return ivs
        except Exception:
            return []

    def add_iv_to_history(self, contract_id: str, call_iv: Optional[float],
                          put_iv: Optional[float], timestamp: Optional[float] = None):
        """
        Add IV values to history.

        Args:
            contract_id: Unique contract identifier
            call_iv: Call implied volatility
            put_iv: Put implied volatility
            timestamp: Timestamp (defaults to now)
        """
        if not self.redis:
            return

        if timestamp is None:
            timestamp = time.time()

        self.stats['iv_history_writes'] += 1

        # Store call IV
        if call_iv is not None:
            key_call = f"iv_history:{contract_id}:call"
            try:
                self.redis.zadd(key_call, {f"{timestamp}:{call_iv}": timestamp})
                # Keep only recent entries
                self.redis.zremrangebyrank(key_call, 0, -(self.iv_history_depth + 1))
                # Set expiry
                self.redis.expire(key_call, self.state_ttl_seconds)
            except Exception:
                pass

        # Store put IV
        if put_iv is not None:
            key_put = f"iv_history:{contract_id}:put"
            try:
                self.redis.zadd(key_put, {f"{timestamp}:{put_iv}": timestamp})
                # Keep only recent entries
                self.redis.zremrangebyrank(key_put, 0, -(self.iv_history_depth + 1))
                # Set expiry
                self.redis.expire(key_put, self.state_ttl_seconds)
            except Exception:
                pass

    def get_smoothed_iv(self, contract_id: str, current_iv: float,
                        option_type: str = 'call', alpha: float = 0.3) -> float:
        """
        Get IV smoothed with exponential moving average.

        Args:
            contract_id: Unique contract identifier
            current_iv: Current computed IV
            option_type: 'call' or 'put'
            alpha: Smoothing factor (0-1, higher = more weight on current)

        Returns:
            Smoothed IV
        """
        history = self.get_iv_history(contract_id, option_type, limit=1)

        if not history:
            return current_iv

        # Exponential moving average
        previous_iv = history[0]
        smoothed = alpha * current_iv + (1 - alpha) * previous_iv

        return smoothed

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with statistics
        """
        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_accesses if total_accesses > 0 else 0

        return {
            'cache_size': len(self.cache),
            'cache_max_size': self.cache_size,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': hit_rate,
            'redis_reads': self.stats['redis_reads'],
            'redis_writes': self.stats['redis_writes'],
            'iv_history_writes': self.stats['iv_history_writes']
        }

    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()

    def _add_to_cache(self, contract_id: str, state: ContractState):
        """
        Add to LRU cache, evicting oldest if full.

        Args:
            contract_id: Unique contract identifier
            state: Contract state
        """
        # Add/update
        if contract_id in self.cache:
            self.cache.move_to_end(contract_id)
        self.cache[contract_id] = state

        # Evict oldest if cache full
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _load_from_redis(self, contract_id: str) -> Optional[ContractState]:
        """
        Load contract state from Redis.

        Args:
            contract_id: Unique contract identifier

        Returns:
            ContractState if found, None otherwise
        """
        key = f"contract:{contract_id}"

        try:
            state_json = self.redis.get(key)
            if state_json:
                self.stats['redis_reads'] += 1
                data = json.loads(state_json)
                return ContractState.from_dict(data)
        except Exception:
            pass

        return None

    def _save_to_redis(self, contract_id: str, state: ContractState):
        """
        Save contract state to Redis.

        Args:
            contract_id: Unique contract identifier
            state: Contract state
        """
        key = f"contract:{contract_id}"

        try:
            state_json = json.dumps(state.to_dict())
            self.redis.setex(key, self.state_ttl_seconds, state_json)
            self.stats['redis_writes'] += 1
        except Exception:
            pass
