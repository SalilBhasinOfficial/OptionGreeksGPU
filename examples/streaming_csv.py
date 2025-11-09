"""
Example 1: CSV Batch Processing

Demonstrates:
- Loading option data from CSV file
- Processing in batches with JAX
- Computing Greeks for all contracts
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.adapters.csv_adapter import CSVAdapter, create_sample_csv
from streaming.state_manager import StateManager
from streaming.processor import StreamProcessor
import tempfile


def main():
    print("=" * 60)
    print("CSV Batch Processing Example")
    print("=" * 60)

    # Create sample CSV file
    print("\n1. Creating sample CSV file...")
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_file.close()

    create_sample_csv(temp_file.name, num_contracts=500)
    print(f"   Created CSV with 500 contracts: {temp_file.name}")

    # Initialize components
    print("\n2. Initializing components...")
    csv_adapter = CSVAdapter(temp_file.name, batch_size=100)
    state_manager = StateManager(redis_client=None, cache_size=1000)
    processor = StreamProcessor(
        state_manager=state_manager,
        interest_rate=5.0
    )
    print("   ✓ CSV Adapter")
    print("   ✓ State Manager (in-memory only)")
    print("   ✓ Stream Processor (JAX backend)")

    # Process batches
    print("\n3. Processing batches...")
    total_processed = 0
    total_batches = 0

    for batch in csv_adapter.stream():
        results, metrics = processor.process_batch(batch)
        total_processed += len(results)
        total_batches += 1

        # Calculate throughput
        throughput = (len(batch) / metrics.processing_time_ms * 1000) if metrics.processing_time_ms > 0 else 0

        print(f"\n   Batch {total_batches}:")
        print(f"   - Contracts: {len(batch)}")
        print(f"   - JAX compute time: {metrics.jax_compute_time_ms:.2f} ms")
        print(f"   - Total time: {metrics.processing_time_ms:.2f} ms")
        print(f"   - Throughput: {throughput:.0f} updates/sec")

        # Show sample result
        if results:
            sample = results[0]
            print(f"\n   Sample Result ({sample.contract_id}):")
            print(f"   - Underlying: ₹{sample.underlying_price:.2f}")
            print(f"   - Call IV: {sample.call.get('iv', 0)*100:.2f}%")
            print(f"   - Call Delta: {sample.call.get('delta', 0):.4f}")
            print(f"   - Call Gamma: {sample.call.get('gamma', 0):.6f}")
            print(f"   - Put IV: {sample.put.get('iv', 0)*100:.2f}%")
            print(f"   - Put Delta: {sample.put.get('delta', 0):.4f}")

    # Final statistics
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)

    stats = processor.get_stats()
    print(f"\nStatistics:")
    print(f"  Total batches: {stats['batches_processed']}")
    print(f"  Total contracts: {stats['total_updates']}")
    avg_batch = stats['total_updates'] / stats['batches_processed'] if stats['batches_processed'] > 0 else 0
    print(f"  Average batch size: {avg_batch:.1f}")
    print(f"  Average JAX time: {stats['avg_jax_time_ms']:.2f} ms")
    print(f"  Average processing time: {stats['avg_processing_time_ms']:.2f} ms")
    print(f"  Overall throughput: {stats['throughput_updates_per_sec']:.0f} updates/sec")

    cache_stats = state_manager.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Cache size: {cache_stats['cache_size']}/{cache_stats['cache_max_size']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['cache_hit_rate']*100:.1f}%")

    # Cleanup
    os.unlink(temp_file.name)
    print("\n✓ Cleaned up temporary file")


if __name__ == '__main__':
    main()
