"""
Main entry point for streaming option Greeks computation.

This orchestrates all components:
- Data ingestion from configured sources
- Stream processing with JAX
- Output publishing to multiple sinks
"""

import time
import signal
import sys
from typing import Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .config import Config
from .state_manager import StateManager
from .processor import StreamProcessor
from .output.publishers import MultiSinkPublisher


class StreamingGreeksEngine:
    """
    Main streaming engine that coordinates all components.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize streaming engine.

        Args:
            config_file: Path to configuration file (optional)
        """
        # Load configuration
        self.config = Config(config_file)

        # Initialize Redis client
        self.redis_client = None
        if REDIS_AVAILABLE:
            self.redis_client = redis.Redis(
                host=self.config.get('redis.host', 'localhost'),
                port=self.config.get('redis.port', 6379),
                db=self.config.get('redis.db', 0),
                decode_responses=False
            )

        # Initialize state manager
        self.state_manager = StateManager(
            redis_client=self.redis_client,
            cache_size=self.config.get('state.cache_size', 10000),
            iv_history_depth=self.config.get('state.iv_history_depth', 100),
            state_ttl_days=self.config.get('state.state_ttl_days', 7)
        )

        # Initialize stream processor
        self.processor = StreamProcessor(
            state_manager=self.state_manager,
            interest_rate=self.config.get('processing.interest_rate', 5.0),
            use_iv_smoothing=self.config.get('processing.use_iv_smoothing', True),
            smoothing_alpha=self.config.get('processing.smoothing_alpha', 0.3)
        )

        # Initialize publisher
        self.publisher = MultiSinkPublisher(
            redis_client=self.redis_client,
            output_stream_key=self.config.get('output.redis_streams.key', 'greeks:updates'),
            enable_redis_streams=self.config.get('output.redis_streams.enabled', True),
            enable_redis_timeseries=self.config.get('output.redis_timeseries.enabled', False),
            enable_questdb=False  # QuestDB adapter needs separate initialization
        )

        # Initialize adapters
        self.adapters = []
        self._initialize_adapters()

        # Control flags
        self.running = False
        self.shutdown_requested = False

    def _initialize_adapters(self):
        """Initialize configured data source adapters."""
        # CSV adapter
        if self.config.get('sources.csv.enabled', False):
            from .adapters.csv_adapter import CSVAdapter
            adapter = CSVAdapter(
                file_path=self.config.get('sources.csv.file_path', 'data.csv'),
                watch_mode=self.config.get('sources.csv.watch_mode', False),
                batch_size=self.config.get('sources.csv.batch_size', 1000)
            )
            self.adapters.append(('csv', adapter))

        # Redis Streams adapter
        if self.config.get('sources.redis_streams.enabled', False) and self.redis_client:
            from .adapters.redis_streams import RedisStreamsAdapter
            adapter = RedisStreamsAdapter(
                redis_client=self.redis_client,
                stream_key=self.config.get('redis.streams.input_key', 'market:updates'),
                consumer_group=self.config.get('redis.streams.consumer_group', 'greeks_compute'),
                consumer_name=self.config.get('redis.streams.consumer_name', 'worker_1'),
                block_ms=self.config.get('redis.streams.block_ms', 1000),
                count=self.config.get('redis.streams.count', 100)
            )
            self.adapters.append(('redis_streams', adapter))

        # Redis HKEY adapter
        if self.config.get('sources.redis_hkey.enabled', False) and self.redis_client:
            from .adapters.redis_hkey import RedisHKEYAdapter
            adapter = RedisHKEYAdapter(
                redis_client=self.redis_client,
                key_pattern=self.config.get('redis.hkey.key_pattern', 'nse:options:*'),
                poll_interval_ms=self.config.get('redis.hkey.poll_interval_ms', 100)
            )
            self.adapters.append(('redis_hkey', adapter))

    def start(self):
        """
        Start the streaming engine.
        """
        print(f"Starting {self.config.get('app.name')} v{self.config.get('app.version')}")
        print(f"Configured {len(self.adapters)} data source(s)")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.running = True

        # Process from each adapter
        for adapter_name, adapter in self.adapters:
            print(f"Processing from {adapter_name}...")
            self._process_adapter(adapter_name, adapter)

    def _process_adapter(self, name: str, adapter):
        """
        Process updates from an adapter.

        Args:
            name: Adapter name
            adapter: Adapter instance
        """
        try:
            adapter.connect()
            print(f"{name}: Connected successfully")

            # Stream updates
            for batch in adapter.stream():
                if self.shutdown_requested:
                    break

                if not batch:
                    continue

                # Process batch
                start_time = time.time()
                results, metrics = self.processor.process_batch(batch)
                processing_time = (time.time() - start_time) * 1000

                # Publish results
                if results:
                    publish_start = time.time()
                    self.publisher.publish(results)
                    publish_time = (time.time() - publish_start) * 1000

                    # Log stats
                    if self.config.get('monitoring.enabled', True):
                        print(f"{name}: Processed {len(batch)} updates → "
                              f"{len(results)} results in {processing_time:.2f}ms "
                              f"(JAX: {metrics.jax_compute_time_ms:.2f}ms, "
                              f"Publish: {publish_time:.2f}ms)")

        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            adapter.disconnect()
            print(f"{name}: Disconnected")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\nShutdown requested...")
        self.shutdown_requested = True
        self.stop()

    def stop(self):
        """Stop the streaming engine."""
        print("Stopping streaming engine...")
        self.running = False

        # Flush any remaining data
        if self.publisher:
            self.publisher.close()

        # Print final statistics
        self._print_stats()

        print("Shutdown complete")
        sys.exit(0)

    def _print_stats(self):
        """Print system statistics."""
        print("\n" + "="*70)
        print("STREAMING STATISTICS")
        print("="*70)

        # Processor stats
        proc_stats = self.processor.get_stats()
        print(f"\nProcessor:")
        print(f"  Batches processed: {proc_stats['batches_processed']}")
        print(f"  Total updates: {proc_stats['total_updates']}")
        print(f"  Avg processing time: {proc_stats.get('avg_processing_time_ms', 0):.2f}ms")
        print(f"  Avg JAX time: {proc_stats.get('avg_jax_time_ms', 0):.2f}ms")
        print(f"  Throughput: {proc_stats.get('throughput_updates_per_sec', 0):.0f} updates/sec")

        # State manager stats
        state_stats = self.state_manager.get_cache_stats()
        print(f"\nState Manager:")
        print(f"  Cache size: {state_stats['cache_size']}/{state_stats['cache_max_size']}")
        print(f"  Cache hit rate: {state_stats['cache_hit_rate']*100:.1f}%")
        print(f"  Redis reads: {state_stats['redis_reads']}")
        print(f"  Redis writes: {state_stats['redis_writes']}")

        # Publisher stats
        pub_stats = self.publisher.get_stats()
        print(f"\nPublisher:")
        print(f"  Total published: {pub_stats['total_published']}")
        print(f"  Redis Streams: {pub_stats['redis_streams_published']}")
        print(f"  Redis TimeSeries: {pub_stats['redis_timeseries_published']}")
        print(f"  QuestDB: {pub_stats['questdb_published']}")

        print("="*70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Streaming Option Greeks Computation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    args = parser.parse_args()

    # Create and start engine
    engine = StreamingGreeksEngine(config_file=args.config)
    engine.start()


if __name__ == '__main__':
    main()
