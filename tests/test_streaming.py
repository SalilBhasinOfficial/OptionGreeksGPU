"""
Comprehensive test suite for streaming option Greeks computation.

Tests all components:
- Data models
- State manager
- Adapters
- Stream processor
- Output publishers
"""

import unittest
import tempfile
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import streaming components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.models import MarketUpdate, ContractState, OptionType, DataSource
from streaming.state_manager import StateManager
from streaming.processor import StreamProcessor
from streaming.adapters.csv_adapter import CSVAdapter, create_sample_csv


class TestMarketUpdate(unittest.TestCase):
    """Test MarketUpdate data model."""

    def test_create_market_update(self):
        """Test creating a MarketUpdate."""
        update = MarketUpdate(
            contract_id="NIFTY:18000:2025-01-30:CE",
            symbol="NIFTY",
            strike=18000,
            expiry=datetime(2025, 1, 30),
            option_type=OptionType.CALL,
            underlying_price=18050.0,
            call_price=125.5,
            timestamp=time.time(),
            source=DataSource.CSV
        )

        self.assertEqual(update.symbol, "NIFTY")
        self.assertEqual(update.strike, 18000)
        self.assertIsNotNone(update.call_price)

    def test_to_dict(self):
        """Test MarketUpdate serialization."""
        update = MarketUpdate(
            contract_id="NIFTY:18000:2025-01-30:CE",
            symbol="NIFTY",
            strike=18000,
            expiry=datetime(2025, 1, 30),
            option_type=OptionType.CALL,
            underlying_price=18050.0,
            timestamp=time.time(),
            source=DataSource.CSV
        )

        data = update.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data['symbol'], "NIFTY")

    def test_from_dict(self):
        """Test MarketUpdate deserialization."""
        data = {
            'contract_id': "NIFTY:18000:2025-01-30:CE",
            'symbol': "NIFTY",
            'strike': 18000,
            'expiry': "2025-01-30",
            'option_type': "CE",
            'underlying_price': 18050.0,
            'call_price': 125.5,
            'timestamp': time.time(),
            'source': "csv"
        }

        update = MarketUpdate.from_dict(data)
        self.assertEqual(update.symbol, "NIFTY")
        self.assertEqual(update.strike, 18000)


class TestStateManager(unittest.TestCase):
    """Test StateManager."""

    def setUp(self):
        """Set up test state manager (no Redis)."""
        self.state_manager = StateManager(redis_client=None, cache_size=100)

    def test_get_nonexistent_state(self):
        """Test getting state for non-existent contract."""
        state = self.state_manager.get_contract_state("NONEXISTENT")
        self.assertIsNone(state)

    def test_update_contract_state(self):
        """Test updating contract state."""
        state = ContractState(
            contract_id="TEST:100:2025-01-30:CE",
            symbol="TEST",
            strike=100,
            expiry=datetime(2025, 1, 30),
            option_type=OptionType.CALL,
            underlying_price=105.0
        )

        self.state_manager.update_contract_state("TEST:100:2025-01-30:CE", state)

        # Retrieve state
        retrieved = self.state_manager.get_contract_state("TEST:100:2025-01-30:CE")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, "TEST")

    def test_cache_lru_eviction(self):
        """Test LRU cache eviction."""
        # Create state manager with small cache
        sm = StateManager(redis_client=None, cache_size=3)

        # Add 4 states (should evict first one)
        for i in range(4):
            state = ContractState(
                contract_id=f"TEST{i}",
                symbol="TEST",
                strike=100,
                expiry=datetime(2025, 1, 30),
                option_type=OptionType.CALL,
                underlying_price=105.0
            )
            sm.update_contract_state(f"TEST{i}", state)

        # First state should be evicted
        self.assertEqual(len(sm.cache), 3)
        self.assertNotIn("TEST0", sm.cache)

    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.state_manager.get_cache_stats()
        self.assertIn('cache_size', stats)
        self.assertIn('cache_hit_rate', stats)


class TestCSVAdapter(unittest.TestCase):
    """Test CSV adapter."""

    def setUp(self):
        """Set up test CSV file."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.close()

        # Create sample CSV
        create_sample_csv(self.temp_file.name, num_contracts=50)

    def tearDown(self):
        """Clean up test file."""
        os.unlink(self.temp_file.name)

    def test_connect(self):
        """Test CSV adapter connection."""
        adapter = CSVAdapter(self.temp_file.name)
        self.assertTrue(adapter.connect())

    def test_read_batch(self):
        """Test reading batch from CSV."""
        adapter = CSVAdapter(self.temp_file.name)
        updates = adapter.read_batch(limit=10)

        self.assertEqual(len(updates), 10)
        self.assertIsInstance(updates[0], MarketUpdate)

    def test_stream_mode(self):
        """Test streaming from CSV (one-time)."""
        adapter = CSVAdapter(self.temp_file.name, watch_mode=False, batch_size=20)

        batches = []
        for batch in adapter.stream():
            batches.append(batch)
            if len(batches) >= 3:  # Read first 3 batches
                break

        self.assertGreater(len(batches), 0)
        self.assertLessEqual(len(batches[0]), 20)


class TestStreamProcessor(unittest.TestCase):
    """Test stream processor."""

    def setUp(self):
        """Set up test processor."""
        self.state_manager = StateManager(redis_client=None, cache_size=1000)
        self.processor = StreamProcessor(
            state_manager=self.state_manager,
            interest_rate=5.0
        )

    def test_process_empty_batch(self):
        """Test processing empty batch."""
        results, metrics = self.processor.process_batch([])
        self.assertEqual(len(results), 0)

    def test_process_single_update(self):
        """Test processing single market update."""
        update = MarketUpdate(
            contract_id="NIFTY:18000:2025-01-30:CE",
            symbol="NIFTY",
            strike=18000,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL,
            underlying_price=18050.0,
            call_price=125.5,
            put_price=45.0,
            timestamp=time.time(),
            source=DataSource.CSV
        )

        results, metrics = self.processor.process_batch([update])

        self.assertEqual(len(results), 1)
        self.assertIn('iv', results[0].call)
        self.assertIn('delta', results[0].call)

    def test_process_batch(self):
        """Test processing batch of updates."""
        updates = []
        for i in range(10):
            update = MarketUpdate(
                contract_id=f"NIFTY:{18000+i*50}:2025-01-30:CE",
                symbol="NIFTY",
                strike=18000 + i * 50,
                expiry=datetime.now() + timedelta(days=30),
                option_type=OptionType.CALL,
                underlying_price=18050.0,
                call_price=125.5 - i * 5,
                put_price=45.0 + i * 3,
                timestamp=time.time(),
                source=DataSource.CSV
            )
            updates.append(update)

        results, metrics = self.processor.process_batch(updates)

        self.assertEqual(len(results), 10)
        self.assertGreater(metrics.jax_compute_time_ms, 0)

    def test_get_stats(self):
        """Test getting processor statistics."""
        stats = self.processor.get_stats()
        self.assertIn('batches_processed', stats)
        self.assertIn('total_updates', stats)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        """Set up for end-to-end test."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.close()
        create_sample_csv(self.temp_file.name, num_contracts=100)

    def tearDown(self):
        """Clean up."""
        os.unlink(self.temp_file.name)

    def test_csv_to_processor_flow(self):
        """Test complete flow: CSV → Processor → Results."""
        # Create components
        adapter = CSVAdapter(self.temp_file.name, batch_size=50)
        state_manager = StateManager(redis_client=None)
        processor = StreamProcessor(state_manager=state_manager)

        # Process batches
        total_results = 0
        for batch in adapter.stream():
            results, metrics = processor.process_batch(batch)
            total_results += len(results)

            if total_results >= 100:
                break

        self.assertGreaterEqual(total_results, 50)

        # Check stats
        stats = processor.get_stats()
        self.assertGreater(stats['batches_processed'], 0)
        self.assertGreater(stats['total_updates'], 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
