"""
Example 3: Complete Streaming Pipeline

Demonstrates:
- Using configuration file
- Multi-source data ingestion (CSV + Redis)
- State management with IV history
- Multi-sink output (Redis Streams + TimeSeries)
- Full production-ready pipeline

Prerequisites:
- Redis server running (optional, will work without)
- Config file: config/streaming.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import signal
import tempfile
from typing import Optional
from streaming.main import StreamingGreeksEngine
from streaming.adapters.csv_adapter import create_sample_csv


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    print("\n\nShutdown requested... cleaning up...")
    shutdown_requested = True


def main():
    print("=" * 70)
    print("Complete Streaming Pipeline Example")
    print("=" * 70)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for config file
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config',
        'streaming.yaml'
    )

    if not os.path.exists(config_file):
        print(f"\nWARNING: Config file not found: {config_file}")
        print("Using default configuration...")
        config_file = None
    else:
        print(f"\n✓ Using config file: {config_file}")

    # Create sample data file
    print("\n1. Creating sample data...")
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.close()
    create_sample_csv(temp_file.name, num_contracts=1000)
    print(f"   ✓ Created CSV with 1000 contracts: {temp_file.name}")

    # Initialize engine
    print("\n2. Initializing streaming engine...")
    try:
        engine = StreamingGreeksEngine(config_file=config_file)
        print("   ✓ Engine initialized")
    except Exception as e:
        print(f"   ERROR: Failed to initialize engine: {e}")
        print("\n   Note: Redis components will be disabled if Redis is not available")
        print("         The engine will still work with CSV input and in-memory state")
        # Continue anyway
        engine = StreamingGreeksEngine(config_file=None)

    # Show configuration
    print("\n3. Configuration:")
    print(f"   Processing:")
    print(f"   - Batch size: {engine.config.get('processing.batch_size', 500)}")
    print(f"   - Window (ms): {engine.config.get('processing.window_ms', 50)}")
    print(f"   - Interest rate: {engine.config.get('processing.interest_rate', 5.0)}%")
    print(f"\n   State Management:")
    print(f"   - Cache size: {engine.config.get('state.cache_size', 10000)}")
    print(f"   - IV history depth: {engine.config.get('state.iv_history_depth', 100)}")
    print(f"\n   Data Sources:")
    print(f"   - CSV: {engine.config.get('sources.csv.enabled', True)}")
    print(f"   - Redis Streams: {engine.config.get('sources.redis_streams.enabled', False)}")
    print(f"\n   Output Sinks:")
    print(f"   - Redis Streams: {engine.config.get('outputs.redis_streams.enabled', False)}")
    print(f"   - Redis TimeSeries: {engine.config.get('outputs.redis_timeseries.enabled', False)}")

    # Add CSV source
    print("\n4. Adding data source...")
    from streaming.adapters.csv_adapter import CSVAdapter
    csv_adapter = CSVAdapter(
        temp_file.name,
        batch_size=engine.config.get('processing.batch_size', 500)
    )
    engine.add_source(csv_adapter)
    print(f"   ✓ Added CSV adapter: {temp_file.name}")

    # Start processing
    print("\n5. Starting streaming pipeline...")
    print("   Press Ctrl+C to stop\n")
    print("-" * 70)

    try:
        # Run for a limited time or until data exhausted
        total_processed = 0
        total_batches = 0
        start_time = time.time()
        last_stats_time = start_time

        # Process batches
        for batch_results in engine.process_stream():
            if shutdown_requested:
                break

            total_batches += 1
            batch_size = len(batch_results)
            total_processed += batch_size

            # Print progress every 5 batches or 5 seconds
            current_time = time.time()
            if total_batches % 5 == 0 or (current_time - last_stats_time) >= 5:
                elapsed = current_time - start_time
                throughput = total_processed / elapsed if elapsed > 0 else 0

                print(f"\nBatch {total_batches}:")
                print(f"  - Processed: {batch_size} contracts")
                print(f"  - Total: {total_processed} contracts")
                print(f"  - Elapsed: {elapsed:.1f}s")
                print(f"  - Throughput: {throughput:.0f} updates/sec")

                # Show sample result
                if batch_results:
                    sample = batch_results[0]
                    print(f"\n  Sample: {sample.contract_id}")
                    print(f"  - Underlying: ₹{sample.underlying_price:.2f}")
                    print(f"  - Call IV: {sample.call.get('iv', 0)*100:.2f}%")
                    print(f"  - Call Delta: {sample.call.get('delta', 0):.4f}")
                    print(f"  - Call Gamma: {sample.call.get('gamma', 0):.6f}")
                    print(f"  - Put IV: {sample.put.get('iv', 0)*100:.2f}%")
                    print(f"  - Put Delta: {sample.put.get('delta', 0):.4f}")

                last_stats_time = current_time

            # Stop after processing all data from CSV
            if total_processed >= 1000:
                print("\n✓ Processed all available data")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()

    # Shutdown
    print("\n" + "=" * 70)
    print("Shutting down...")
    print("=" * 70)

    engine.shutdown()
    print("   ✓ Engine shutdown complete")

    # Final statistics
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)

    stats = engine.get_stats()

    print(f"\nProcessing:")
    print(f"  Total batches: {stats['processor']['batches_processed']}")
    print(f"  Total updates: {stats['processor']['total_updates']}")
    print(f"  Average batch size: {stats['processor'].get('avg_batch_size', 0):.1f}")
    print(f"  Average JAX time: {stats['processor'].get('avg_jax_time_ms', 0):.2f} ms")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Overall throughput: {total_processed / (total_time or 1):.0f} updates/sec")

    print(f"\nState Management:")
    cache_stats = stats['state_manager']
    print(f"  Cache size: {cache_stats['cache_size']}/{cache_stats['cache_max_size']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats.get('cache_hit_rate', 0)*100:.1f}%")

    if 'publisher' in stats:
        print(f"\nOutput Publishing:")
        pub_stats = stats['publisher']
        print(f"  Total published: {pub_stats['total_published']}")
        if pub_stats.get('redis_streams_published', 0) > 0:
            print(f"  Redis Streams: {pub_stats['redis_streams_published']}")
        if pub_stats.get('redis_timeseries_published', 0) > 0:
            print(f"  Redis TimeSeries: {pub_stats['redis_timeseries_published']}")
        if pub_stats.get('questdb_published', 0) > 0:
            print(f"  QuestDB: {pub_stats['questdb_published']}")

    # Cleanup
    os.unlink(temp_file.name)
    print("\n✓ Cleaned up temporary files")

    print("\n" + "=" * 70)
    print("Example Complete")
    print("=" * 70)

    print("\nKey Features Demonstrated:")
    print("  ✓ Configuration-based setup")
    print("  ✓ CSV data ingestion")
    print("  ✓ Batch processing with JAX")
    print("  ✓ State management with LRU cache")
    print("  ✓ IV history tracking")
    print("  ✓ Graceful shutdown")
    print("  ✓ Performance metrics")

    print("\nNext Steps:")
    print("  - Enable Redis for state persistence and output")
    print("  - Add more data sources (Redis Streams, HKEY, TimeSeries)")
    print("  - Configure multi-sink output (TimeSeries, QuestDB)")
    print("  - Scale horizontally with multiple processors")


if __name__ == '__main__':
    main()
