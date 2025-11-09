"""
Example 2: Redis Streams Real-time Processing

Demonstrates:
- Reading from Redis Streams with consumer groups
- Real-time Greeks computation
- State management with Redis persistence
- Publishing results to Redis Streams

Prerequisites:
- Redis server running on localhost:6379
- Install: pip install redis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import redis
from streaming.adapters.redis_streams import RedisStreamsAdapter
from streaming.state_manager import StateManager
from streaming.processor import StreamProcessor
from streaming.output.publishers import MultiSinkPublisher
from streaming.models import MarketUpdate, OptionType, DataSource
from datetime import datetime, timedelta


def setup_redis_connection():
    """Connect to Redis server."""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        return r
    except redis.ConnectionError:
        print("ERROR: Cannot connect to Redis server")
        print("Please ensure Redis is running on localhost:6379")
        print("\nTo start Redis:")
        print("  - Linux/Mac: redis-server")
        print("  - Docker: docker run -d -p 6379:6379 redis:latest")
        return None


def publish_sample_data(redis_client, stream_key: str, num_updates: int = 100):
    """Publish sample market updates to Redis Stream."""
    print(f"\nPublishing {num_updates} sample updates to Redis Stream '{stream_key}'...")

    for i in range(num_updates):
        # Create sample market update
        symbol = ['NIFTY', 'BANKNIFTY', 'FINNIFTY'][i % 3]
        base_price = {'NIFTY': 18000, 'BANKNIFTY': 42000, 'FINNIFTY': 19000}[symbol]
        strike = base_price + (i % 10 - 5) * 50

        data = {
            'contract_id': f"{symbol}:{strike}:2025-01-30:CE",
            'symbol': symbol,
            'strike': str(strike),
            'expiry': '2025-01-30',
            'option_type': 'CE',
            'underlying_price': str(base_price + (i % 10) * 10),
            'call_price': str(max(10, 150 - i * 2)),
            'put_price': str(max(10, 50 + i)),
            'timestamp': str(time.time()),
            'source': 'redis_stream'
        }

        redis_client.xadd(stream_key, data)

        # Simulate real-time with delay
        if (i + 1) % 10 == 0:
            print(f"  Published {i + 1}/{num_updates} updates...")
            time.sleep(0.1)

    print(f"  ✓ Published all {num_updates} updates")


def main():
    print("=" * 60)
    print("Redis Streams Real-time Processing Example")
    print("=" * 60)

    # Connect to Redis
    print("\n1. Connecting to Redis...")
    redis_client = setup_redis_connection()
    if not redis_client:
        return

    print("   ✓ Connected to Redis")

    # Configuration
    input_stream_key = "nse:market:updates"
    output_stream_key = "greeks:updates"
    consumer_group = "greeks_processors"
    consumer_name = "processor_1"

    # Create consumer group (ignore error if exists)
    try:
        redis_client.xgroup_create(input_stream_key, consumer_group, id='0', mkstream=True)
        print(f"   ✓ Created consumer group '{consumer_group}'")
    except redis.ResponseError:
        print(f"   ✓ Consumer group '{consumer_group}' already exists")

    # Publish sample data
    publish_sample_data(redis_client, input_stream_key, num_updates=100)

    # Initialize components
    print("\n2. Initializing streaming components...")
    adapter = RedisStreamsAdapter(
        redis_client=redis_client,
        stream_key=input_stream_key,
        consumer_group=consumer_group,
        consumer_name=consumer_name,
        count=50,
        block_ms=1000
    )

    state_manager = StateManager(
        redis_client=redis_client,
        cache_size=1000,
        iv_history_depth=100
    )

    processor = StreamProcessor(
        state_manager=state_manager,
        interest_rate=5.0
    )

    publisher = MultiSinkPublisher(
        redis_client=redis_client,
        output_stream_key=output_stream_key,
        enable_redis_streams=True,
        enable_redis_timeseries=False,  # Disabled for this example
        enable_questdb=False
    )

    print("   ✓ Redis Streams Adapter")
    print("   ✓ State Manager (with Redis persistence)")
    print("   ✓ Stream Processor (JAX backend)")
    print("   ✓ Multi-sink Publisher")

    # Process stream
    print("\n3. Processing stream (will read up to 100 updates)...")
    total_processed = 0
    total_batches = 0
    start_time = time.time()

    try:
        for batch in adapter.stream():
            if not batch:
                continue

            # Process batch
            results, metrics = processor.process_batch(batch)
            total_processed += len(results)
            total_batches += 1

            # Publish results
            publisher.publish(results)

            # Calculate throughput
            throughput = (len(batch) / metrics.processing_time_ms * 1000) if metrics.processing_time_ms > 0 else 0

            print(f"\n   Batch {total_batches}:")
            print(f"   - Updates: {len(batch)}")
            print(f"   - JAX time: {metrics.jax_compute_time_ms:.2f} ms")
            print(f"   - Total time: {metrics.processing_time_ms:.2f} ms")
            print(f"   - Throughput: {throughput:.0f} updates/sec")

            # Show sample result
            if results:
                sample = results[0]
                print(f"\n   Sample: {sample.contract_id}")
                print(f"   - Call IV: {sample.call.get('iv', 0)*100:.2f}%")
                print(f"   - Call Delta: {sample.call.get('delta', 0):.4f}")
                print(f"   - Put Delta: {sample.put.get('delta', 0):.4f}")

            # Stop after processing all published data
            if total_processed >= 100:
                print("\n   ✓ Processed all updates, stopping...")
                break

    except KeyboardInterrupt:
        print("\n\n   Interrupted by user")

    elapsed_time = time.time() - start_time

    # Final statistics
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)

    stats = processor.get_stats()
    print(f"\nProcessor Statistics:")
    print(f"  Total batches: {stats['batches_processed']}")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Average batch size: {stats['avg_batch_size']:.1f}")
    print(f"  Average JAX time: {stats['avg_jax_time_ms']:.2f} ms")
    print(f"  Overall throughput: {total_processed / (elapsed_time or 1):.0f} updates/sec")

    cache_stats = state_manager.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Cache size: {cache_stats['cache_size']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['cache_hit_rate']*100:.1f}%")

    pub_stats = publisher.get_stats()
    print(f"\nPublisher Statistics:")
    print(f"  Total published: {pub_stats['total_published']}")
    print(f"  Redis Streams: {pub_stats['redis_streams_published']}")

    # Check output stream
    output_count = redis_client.xlen(output_stream_key)
    print(f"\nOutput Stream '{output_stream_key}':")
    print(f"  Messages: {output_count}")

    # Show sample output
    if output_count > 0:
        sample_output = redis_client.xrevrange(output_stream_key, count=1)
        if sample_output:
            msg_id, msg_data = sample_output[0]
            print(f"\n  Sample Output Message:")
            print(f"  - ID: {msg_id}")
            print(f"  - Contract: {msg_data.get('contract_id')}")
            print(f"  - Call IV: {msg_data.get('call_iv')}")
            print(f"  - Call Delta: {msg_data.get('call_delta')}")

    print("\n✓ Example complete")
    print("\nNote: Redis data persists. To clean up:")
    print(f"  redis-cli DEL {input_stream_key} {output_stream_key}")


if __name__ == '__main__':
    main()
