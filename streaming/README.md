# Streaming Option Greeks Computation

Real-time and batch processing system for computing option Greeks at scale using JAX automatic differentiation.

## Features

- **High-Performance Computing**: JAX-based Greeks computation with GPU acceleration (42× faster than CPU)
- **Multiple Data Sources**: Ingest from CSV, Redis Streams, Redis HKEY, Redis TimeSeries, QuestDB
- **Micro-batching**: Optimized batch processing (50ms windows, 500 contracts/batch)
- **State Management**: Two-tier storage (LRU cache + Redis) for sub-millisecond lookups
- **IV History**: Track implied volatility history with exponential smoothing
- **Multi-Sink Output**: Publish to Redis Streams, Redis TimeSeries, QuestDB
- **Horizontal Scaling**: Consumer groups support for distributed processing
- **Production Ready**: Configuration-based setup, graceful shutdown, monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                              │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│   CSV    │  Redis   │  Redis   │  Redis   │     QuestDB        │
│  Files   │ Streams  │  HKEY    │   TS     │                    │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴──────┬─────────────┘
     │          │          │          │            │
     └──────────┴──────────┴──────────┴────────────┘
                         │
                    ┌────▼─────┐
                    │ Adapters │
                    └────┬─────┘
                         │
              ┌──────────▼───────────┐
              │  Stream Processor    │
              │  - Micro-batching    │
              │  - JAX Greeks        │
              │  - State enrichment  │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  State Manager       │
              │  - LRU Cache         │
              │  - Redis Store       │
              │  - IV History        │
              └──────────┬───────────┘
                         │
              ┌──────────▼───────────┐
              │  Output Publisher    │
              └──────────┬───────────┘
                         │
     ┌──────────┬────────┴────────┬──────────┐
     │          │                 │          │
┌────▼─────┬───▼────┬───────────▼─────┬────▼──────┐
│  Redis   │ Redis  │    QuestDB      │   REST    │
│ Streams  │   TS   │   (Analytics)   │    API    │
└──────────┴────────┴─────────────────┴───────────┘
```

## Installation

### Prerequisites

```bash
# Core dependencies (required)
pip install jax jaxlib numpy pandas

# Redis support (optional)
pip install redis

# QuestDB support (optional)
pip install psycopg2-binary
```

### GPU Support (Optional)

For GPU acceleration:

```bash
# CUDA 11.x
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 12.x
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Redis Setup (Optional)

For Redis-based features:

```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or install Redis locally
# Linux: sudo apt-get install redis-server
# Mac: brew install redis
```

## Quick Start

### 1. CSV Batch Processing

Process option contracts from CSV file:

```python
from streaming.adapters.csv_adapter import CSVAdapter
from streaming.state_manager import StateManager
from streaming.processor import StreamProcessor

# Initialize components
csv_adapter = CSVAdapter('options.csv', batch_size=500)
state_manager = StateManager(redis_client=None)
processor = StreamProcessor(state_manager=state_manager, interest_rate=5.0)

# Process batches
for batch in csv_adapter.stream():
    results, metrics = processor.process_batch(batch)

    for result in results:
        print(f"{result.contract_id}: IV={result.call['iv']:.2%}")
```

### 2. Redis Streams (Real-time)

Process real-time market updates:

```python
import redis
from streaming.adapters.redis_streams import RedisStreamsAdapter

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Create consumer group
r.xgroup_create('market:updates', 'greeks_processors', id='0', mkstream=True)

# Initialize adapter
adapter = RedisStreamsAdapter(
    redis_client=r,
    stream_key='market:updates',
    consumer_group='greeks_processors',
    consumer_name='processor_1'
)

# Process stream
for batch in adapter.stream():
    results, metrics = processor.process_batch(batch)
    # Handle results...
```

### 3. Complete Pipeline with Configuration

Using YAML configuration:

```python
from streaming.main import StreamingGreeksEngine

# Initialize engine
engine = StreamingGreeksEngine(config_file='config/streaming.yaml')

# Add data source
from streaming.adapters.csv_adapter import CSVAdapter
csv_adapter = CSVAdapter('options.csv')
engine.add_source(csv_adapter)

# Process stream
for results in engine.process_stream():
    # Results are automatically published to configured sinks
    print(f"Processed {len(results)} contracts")

# Shutdown
engine.shutdown()
```

## Configuration

Edit `config/streaming.yaml`:

```yaml
processing:
  batch_size: 500           # Contracts per batch
  window_ms: 50             # Micro-batch window
  interest_rate: 5.0        # Risk-free rate (%)

state:
  cache_size: 10000         # LRU cache size
  iv_history_depth: 100     # IV history length
  state_ttl_days: 7         # Redis TTL

sources:
  csv:
    enabled: true
  redis_streams:
    enabled: true
    stream_key: "market:updates"
    consumer_group: "greeks_processors"
  redis_hkey:
    enabled: false
    key_pattern: "nse:options:*"
    poll_interval_sec: 1.0

outputs:
  redis_streams:
    enabled: true
    stream_key: "greeks:updates"
  redis_timeseries:
    enabled: true
  questdb:
    enabled: false
    host: "localhost"
    port: 9009
```

## Data Models

### Input: MarketUpdate

Market data for a single option contract:

```python
from streaming.models import MarketUpdate, OptionType
from datetime import datetime

update = MarketUpdate(
    contract_id="NIFTY:18000:2025-01-30:CE",
    symbol="NIFTY",
    strike=18000,
    expiry=datetime(2025, 1, 30),
    option_type=OptionType.CALL,
    underlying_price=18050.0,
    call_price=125.5,
    put_price=45.0,
    timestamp=time.time(),
    source=DataSource.CSV
)
```

### Output: GreeksResult

Computed Greeks for both call and put:

```python
result = GreeksResult(
    contract_id="NIFTY:18000:2025-01-30:CE",
    timestamp=time.time(),
    underlying_price=18050.0,
    call={
        'price': 125.5,
        'iv': 0.18,           # Implied volatility
        'delta': 0.53,        # Price sensitivity
        'gamma': 0.0002,      # Delta sensitivity
        'vega': 45.2,         # Volatility sensitivity
        'theta': -12.5,       # Time decay
        'rho': 8.3            # Interest rate sensitivity
    },
    put={
        'price': 45.0,
        'iv': 0.19,
        'delta': -0.47,
        'gamma': 0.0002,
        'vega': 45.2,
        'theta': -8.1,
        'rho': -6.7
    }
)
```

## Data Sources

### CSV Adapter

Batch import or file watching:

```python
from streaming.adapters.csv_adapter import CSVAdapter

# One-time batch
adapter = CSVAdapter('data.csv', watch_mode=False)

# Continuous file watching
adapter = CSVAdapter('data.csv', watch_mode=True, poll_interval_sec=1.0)
```

**CSV Format:**
```csv
symbol,strike,expiry,option_type,underlying_price,call_price,put_price
NIFTY,18000,2025-01-30,CE,18050,125.5,45.0
```

### Redis Streams

Real-time event streaming with consumer groups:

```python
from streaming.adapters.redis_streams import RedisStreamsAdapter

adapter = RedisStreamsAdapter(
    redis_client=redis_client,
    stream_key='market:updates',
    consumer_group='greeks_processors',
    consumer_name='processor_1',
    count=50,              # Batch size
    block_ms=1000          # Block timeout
)
```

**Publishing to Stream:**
```python
redis_client.xadd('market:updates', {
    'contract_id': 'NIFTY:18000:2025-01-30:CE',
    'symbol': 'NIFTY',
    'strike': '18000',
    'underlying_price': '18050',
    'call_price': '125.5',
    # ...
})
```

### Redis HKEY

Polling Redis hashes for snapshots:

```python
from streaming.adapters.redis_hkey import RedisHKEYAdapter

adapter = RedisHKEYAdapter(
    redis_client=redis_client,
    key_pattern='nse:options:*',
    poll_interval_sec=1.0
)
```

**Hash Structure:**
```
HSET nse:options:NIFTY:18000:2025-01-30:CE
  symbol "NIFTY"
  strike "18000"
  underlying_price "18050"
  call_price "125.5"
  put_price "45.0"
```

### Redis TimeSeries

Query time-series data:

```python
from streaming.adapters.redis_timeseries import RedisTimeSeriesAdapter

adapter = RedisTimeSeriesAdapter(
    redis_client=redis_client,
    key_pattern='ts:market:*',
    aggregation_sec=60
)
```

### QuestDB

High-performance time-series database:

```python
from streaming.adapters.questdb import QuestDBAdapter

adapter = QuestDBAdapter(
    host='localhost',
    port=8812,
    table='options_market_data',
    batch_size=10000
)
```

## Output Sinks

### Multi-Sink Publisher

Publish to multiple destinations:

```python
from streaming.output.publishers import MultiSinkPublisher

publisher = MultiSinkPublisher(
    redis_client=redis_client,
    questdb_adapter=questdb_adapter,
    output_stream_key='greeks:updates',
    enable_redis_streams=True,
    enable_redis_timeseries=True,
    enable_questdb=True
)

# Publish results
publisher.publish(results)

# Flush buffered data
publisher.flush_questdb()
```

### Redis Streams Output

Real-time event stream for downstream consumers:

```python
# Results published to 'greeks:updates'
# Read with:
results = redis_client.xread({'greeks:updates': '0-0'}, count=100)
```

### Redis TimeSeries Output

Metrics for monitoring and alerting:

```python
# Time-series keys created automatically:
# ts:greeks:NIFTY:18000:2025-01-30:CE:call_iv
# ts:greeks:NIFTY:18000:2025-01-30:CE:call_delta
# ...

# Query with TS.RANGE
redis_client.execute_command(
    'TS.RANGE',
    'ts:greeks:NIFTY:18000:2025-01-30:CE:call_iv',
    '-', '+',
    'AGGREGATION', 'avg', 60000  # 1-minute average
)
```

### QuestDB Output

SQL analytics database:

```sql
-- Query computed Greeks
SELECT contract_id, timestamp, call_iv, call_delta, put_delta
FROM greeks_results
WHERE timestamp > dateadd('h', -1, now())
ORDER BY timestamp DESC;

-- Calculate average IV by symbol
SELECT symbol, avg(call_iv) as avg_iv
FROM greeks_results
WHERE timestamp > dateadd('d', -1, now())
GROUP BY symbol;
```

## State Management

### Two-Tier Storage

- **Tier 1**: In-memory LRU cache (sub-microsecond)
- **Tier 2**: Redis persistence (sub-millisecond)

```python
from streaming.state_manager import StateManager

state_manager = StateManager(
    redis_client=redis_client,
    cache_size=10000,           # LRU cache size
    iv_history_depth=100,       # IV history length
    state_ttl_days=7            # Redis TTL
)

# Get contract state
state = state_manager.get_contract_state(contract_id)

# Update state
state_manager.update_from_market_update(update, call_greeks, put_greeks)

# Get IV history
iv_history = state_manager.get_iv_history(contract_id, 'call', limit=10)

# Get smoothed IV
smoothed_iv = state_manager.get_smoothed_iv(contract_id, current_iv, 'call')
```

### IV History & Smoothing

Exponential moving average for IV smoothing:

```python
# Alpha = 0.3 (30% weight on current value)
smoothed_iv = 0.3 * current_iv + 0.7 * previous_iv
```

Stored in Redis sorted sets:
```
ZADD iv_history:NIFTY:18000:2025-01-30:CE:call
  1704067200 "1704067200:0.18"
  1704067260 "1704067260:0.19"
  1704067320 "1704067320:0.18"
```

## Performance

### Benchmarks

- **JAX (GPU)**: 9× faster than Numba CUDA
- **JAX (CPU)**: 42× faster than Numba CPU
- **Throughput**: 100,000+ updates/sec
- **Latency**: <5ms per batch (500 contracts)
- **State lookup**: <1ms (cache hit)

### Optimization Tips

1. **Batch Size**: Use 500-1000 for optimal GPU utilization
2. **Cache Size**: Set to 2-3× active contracts
3. **Window Size**: 50-100ms for micro-batching
4. **JIT Warmup**: First batch slower due to XLA compilation
5. **Buffer Writes**: Use QuestDB batching for bulk writes

## Examples

See `examples/` directory:

- **streaming_csv.py**: CSV batch processing
- **streaming_redis.py**: Redis Streams real-time processing
- **streaming_complete.py**: Complete pipeline with configuration

Run examples:

```bash
# CSV example (no dependencies)
python examples/streaming_csv.py

# Redis example (requires Redis)
python examples/streaming_redis.py

# Complete pipeline
python examples/streaming_complete.py
```

## Testing

Run test suite:

```bash
# All tests
python -m pytest tests/test_streaming.py -v

# Specific test class
python -m pytest tests/test_streaming.py::TestStreamProcessor -v

# With coverage
python -m pytest tests/test_streaming.py --cov=streaming --cov-report=html
```

Test categories:
- Data models (MarketUpdate, ContractState)
- State management (LRU cache, Redis persistence)
- Adapters (CSV, Redis Streams)
- Stream processor (JAX integration)
- End-to-end integration

## Monitoring

### Processor Statistics

```python
stats = processor.get_stats()
print(f"Batches: {stats['batches_processed']}")
print(f"Updates: {stats['total_updates']}")
print(f"Avg JAX time: {stats['avg_jax_time_ms']:.2f} ms")
print(f"Throughput: {stats['throughput']:.0f} updates/sec")
```

### Cache Statistics

```python
cache_stats = state_manager.get_cache_stats()
print(f"Cache hit rate: {cache_stats['cache_hit_rate']*100:.1f}%")
print(f"Cache size: {cache_stats['cache_size']}")
```

### Publisher Statistics

```python
pub_stats = publisher.get_stats()
print(f"Published: {pub_stats['total_published']}")
print(f"Errors: {pub_stats['publish_errors']}")
```

## Production Deployment

### Horizontal Scaling

Use Redis consumer groups:

```yaml
# Processor 1
consumer_group: "greeks_processors"
consumer_name: "processor_1"

# Processor 2
consumer_group: "greeks_processors"
consumer_name: "processor_2"
```

### Graceful Shutdown

```python
import signal

def signal_handler(sig, frame):
    print("Shutting down...")
    engine.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### Error Handling

```python
try:
    for batch in adapter.stream():
        try:
            results, metrics = processor.process_batch(batch)
            publisher.publish(results)
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Continue processing
except Exception as e:
    logger.critical(f"Fatal error: {e}")
    engine.shutdown()
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming.log'),
        logging.StreamHandler()
    ]
)
```

## Troubleshooting

### Redis Connection Failed

```
ERROR: Cannot connect to Redis server
```

**Solution**: Ensure Redis is running:
```bash
docker run -d -p 6379:6379 redis:latest
# or
redis-server
```

### JAX GPU Not Available

```
WARNING: No GPU/TPU found, falling back to CPU
```

**Solution**: Install CUDA-enabled JAX:
```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Slow Initial Batch

**Cause**: XLA compilation on first run

**Solution**: This is normal. Subsequent batches will be fast.

### Memory Issues

**Symptoms**: Out of memory errors

**Solutions**:
- Reduce `batch_size` in config
- Reduce `cache_size` in state manager
- Enable Redis for state persistence
- Use streaming mode instead of batch loading

### Invalid IV Results

**Cause**: Option prices violate arbitrage bounds

**Solution**: Validate input data:
- Call price > intrinsic value
- Put price > intrinsic value
- Prices > 0

## API Reference

See individual module docstrings:

- `streaming.models`: Data models
- `streaming.state_manager`: State management
- `streaming.processor`: Stream processing
- `streaming.adapters`: Data source adapters
- `streaming.output`: Output publishers
- `streaming.config`: Configuration
- `streaming.main`: Main engine

## Contributing

When adding new features:

1. Add data models to `models.py`
2. Create adapter in `adapters/`
3. Update processor logic if needed
4. Add tests to `tests/test_streaming.py`
5. Update this README

## License

Same as OptionGreeksGPU main project.

## Support

For issues and questions:
- GitHub Issues: https://github.com/SalilBhasinOfficial/OptionGreeksGPU
- Documentation: See `JAX_IMPLEMENTATION.md` and `STREAMING_ARCHITECTURE.md`
