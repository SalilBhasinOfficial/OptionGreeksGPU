"""
Stream processor with JAX integration for option Greeks computation.

This is the core processing engine that:
1. Receives batches of market updates
2. Enriches with state (IV history, metadata)
3. Computes Greeks using JAX
4. Updates state
5. Publishes results
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .models import MarketUpdate, GreeksResult, ProcessingMetrics, ContractState
from .state_manager import StateManager

# Import JAX Greeks computation
try:
    from OptionGreeksGPU.GreeksJAX import calculate_option_metrics
    JAX_AVAILABLE = True
except ImportError:
    # Fallback to batch mode if JAX not available
    from OptionGreeksGPU.Compute import calculate_option_metrics
    JAX_AVAILABLE = False


class StreamProcessor:
    """
    Main stream processing engine with JAX integration.

    Orchestrates:
    - State management
    - Greeks computation
    - Result publishing
    """

    def __init__(self, state_manager: StateManager,
                 interest_rate: float = 5.0,
                 use_iv_smoothing: bool = True,
                 smoothing_alpha: float = 0.3):
        """
        Initialize stream processor.

        Args:
            state_manager: StateManager instance
            interest_rate: Risk-free interest rate (%)
            use_iv_smoothing: Enable IV smoothing with history
            smoothing_alpha: Smoothing factor for IV (0-1)
        """
        self.state_manager = state_manager
        self.interest_rate = interest_rate
        self.use_iv_smoothing = use_iv_smoothing
        self.smoothing_alpha = smoothing_alpha

        # Processing statistics
        self.stats = {
            'batches_processed': 0,
            'total_updates': 0,
            'total_processing_time_ms': 0.0,
            'total_jax_time_ms': 0.0
        }

    def process_batch(self, updates: List[MarketUpdate]) -> Tuple[List[GreeksResult], ProcessingMetrics]:
        """
        Process a batch of market updates.

        Args:
            updates: List of MarketUpdate objects

        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()

        # Group updates by (symbol, expiry) for efficient computation
        grouped = self._group_by_expiry(updates)

        all_results = []
        total_jax_time = 0.0
        total_state_lookup_time = 0.0
        total_state_update_time = 0.0

        for (symbol, expiry), group_updates in grouped.items():
            # Calculate days to expiry
            days_to_expiry = self._calculate_days_to_expiry(expiry)

            if days_to_expiry <= 0:
                # Skip expired contracts
                continue

            # Enrich with state
            state_start = time.time()
            enriched = self._enrich_with_state(group_updates)
            total_state_lookup_time += (time.time() - state_start) * 1000

            # Prepare JAX input
            jax_input = self._prepare_jax_input(enriched, days_to_expiry)

            # Compute Greeks using JAX
            jax_start = time.time()
            greeks_arrays = calculate_option_metrics(
                option_data=jax_input['option_data'],
                days_to_expiry=days_to_expiry,
                interest_rate=self.interest_rate
            )
            total_jax_time += (time.time() - jax_start) * 1000

            # Parse results and update state
            state_update_start = time.time()
            batch_results = self._parse_and_update_state(enriched, greeks_arrays)
            total_state_update_time += (time.time() - state_update_start) * 1000

            all_results.extend(batch_results)

        # Calculate metrics
        total_time = (time.time() - start_time) * 1000
        metrics = ProcessingMetrics(
            batch_size=len(updates),
            processing_time_ms=total_time,
            jax_compute_time_ms=total_jax_time,
            state_lookup_time_ms=total_state_lookup_time,
            state_update_time_ms=total_state_update_time,
            output_publish_time_ms=0.0,  # Will be updated by output layer
            timestamp=time.time()
        )

        # Update statistics
        self.stats['batches_processed'] += 1
        self.stats['total_updates'] += len(updates)
        self.stats['total_processing_time_ms'] += total_time
        self.stats['total_jax_time_ms'] += total_jax_time

        return all_results, metrics

    def _group_by_expiry(self, updates: List[MarketUpdate]) -> Dict[Tuple[str, datetime], List[MarketUpdate]]:
        """
        Group updates by (symbol, expiry) for batch processing.

        Args:
            updates: List of MarketUpdate objects

        Returns:
            Dictionary mapping (symbol, expiry) to list of updates
        """
        grouped = defaultdict(list)

        for update in updates:
            key = (update.symbol, update.expiry)
            grouped[key].append(update)

        return dict(grouped)

    def _calculate_days_to_expiry(self, expiry: datetime) -> float:
        """
        Calculate days to expiry from now.

        Args:
            expiry: Expiration datetime

        Returns:
            Days to expiry (can be fractional)
        """
        now = datetime.now()

        # If expiry has no time component, assume market close (3:30 PM IST)
        if expiry.hour == 0 and expiry.minute == 0:
            expiry = expiry.replace(hour=15, minute=30)

        delta = expiry - now
        days = delta.total_seconds() / 86400

        return max(0, days)

    def _enrich_with_state(self, updates: List[MarketUpdate]) -> List[Dict]:
        """
        Enrich updates with state from state manager.

        Args:
            updates: List of MarketUpdate objects

        Returns:
            List of enriched dictionaries
        """
        enriched = []

        for update in updates:
            # Get contract state
            state = self.state_manager.get_contract_state(update.contract_id)

            # Get IV history for smoothing
            call_iv_history = self.state_manager.get_iv_history(
                update.contract_id, 'call', limit=10
            )
            put_iv_history = self.state_manager.get_iv_history(
                update.contract_id, 'put', limit=10
            )

            enriched.append({
                'update': update,
                'state': state,
                'call_iv_history': call_iv_history,
                'put_iv_history': put_iv_history
            })

        return enriched

    def _prepare_jax_input(self, enriched: List[Dict], days_to_expiry: float) -> Dict:
        """
        Prepare input for JAX Greeks computation.

        Args:
            enriched: List of enriched update dictionaries
            days_to_expiry: Days to expiration

        Returns:
            Dictionary with JAX input format
        """
        n = len(enriched)
        option_data = np.zeros((n, 6), dtype=np.float32)

        for i, item in enumerate(enriched):
            update = item['update']

            option_data[i, 0] = update.strike
            option_data[i, 1] = update.underlying_price
            option_data[i, 2] = update.call_price if update.call_price else 0.0
            option_data[i, 3] = 0  # Call type flag
            option_data[i, 4] = update.put_price if update.put_price else 0.0
            option_data[i, 5] = 1  # Put type flag

        return {
            'option_data': option_data,
            'days_to_expiry': days_to_expiry,
            'interest_rate': self.interest_rate
        }

    def _parse_and_update_state(self, enriched: List[Dict],
                                greeks_arrays: List[np.ndarray]) -> List[GreeksResult]:
        """
        Parse Greeks results and update state.

        Args:
            enriched: List of enriched update dictionaries
            greeks_arrays: List of 14 arrays from JAX

        Returns:
            List of GreeksResult objects
        """
        # Unpack Greeks arrays
        call_IVs, call_deltas, call_delta2s, call_vegas, call_gammas, \
        call_thetas, call_rhos, put_IVs, put_deltas, put_delta2s, \
        put_vegas, put_gammas, put_thetas, put_rhos = greeks_arrays

        results = []

        for i, item in enumerate(enriched):
            update = item['update']

            # Apply IV smoothing if enabled
            call_iv = float(call_IVs[i])
            put_iv = float(put_IVs[i])

            if self.use_iv_smoothing:
                if item['call_iv_history']:
                    call_iv = self.smoothing_alpha * call_iv + \
                             (1 - self.smoothing_alpha) * item['call_iv_history'][0]
                if item['put_iv_history']:
                    put_iv = self.smoothing_alpha * put_iv + \
                            (1 - self.smoothing_alpha) * item['put_iv_history'][0]

            # Create Greeks dictionaries
            call_greeks = {
                'iv': call_iv,
                'delta': float(call_deltas[i]),
                'gamma': float(call_gammas[i]),
                'vega': float(call_vegas[i]),
                'theta': float(call_thetas[i]),
                'rho': float(call_rhos[i]),
                'delta2': float(call_delta2s[i])
            }

            put_greeks = {
                'iv': put_iv,
                'delta': float(put_deltas[i]),
                'gamma': float(put_gammas[i]),
                'vega': float(put_vegas[i]),
                'theta': float(put_thetas[i]),
                'rho': float(put_rhos[i]),
                'delta2': float(put_delta2s[i])
            }

            # Create result
            result = GreeksResult(
                contract_id=update.contract_id,
                timestamp=update.timestamp,
                underlying_price=update.underlying_price,
                call=call_greeks,
                put=put_greeks
            )

            results.append(result)

            # Update state
            self.state_manager.update_from_market_update(
                update, call_greeks, put_greeks
            )

            # Add to IV history
            self.state_manager.add_iv_to_history(
                update.contract_id, call_iv, put_iv, update.timestamp
            )

        return results

    def get_stats(self) -> Dict:
        """
        Get processor statistics.

        Returns:
            Dictionary with statistics
        """
        avg_processing_time = (
            self.stats['total_processing_time_ms'] / self.stats['batches_processed']
            if self.stats['batches_processed'] > 0 else 0
        )

        avg_jax_time = (
            self.stats['total_jax_time_ms'] / self.stats['batches_processed']
            if self.stats['batches_processed'] > 0 else 0
        )

        return {
            **self.stats,
            'avg_processing_time_ms': avg_processing_time,
            'avg_jax_time_ms': avg_jax_time,
            'throughput_updates_per_sec': (
                self.stats['total_updates'] / (self.stats['total_processing_time_ms'] / 1000)
                if self.stats['total_processing_time_ms'] > 0 else 0
            )
        }
