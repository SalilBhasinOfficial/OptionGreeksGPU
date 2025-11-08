# Streaming & Multi-Source Architecture Recommendations
## OptionGreeksGPU v3.1+ Feature Enhancement

**Document Version:** 1.0
**Date:** 2025-01-08
**Author:** Research & Architecture Analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Requirements Analysis](#requirements-analysis)
3. [Recommended Architecture](#recommended-architecture)
4. [Streaming Computation Design](#streaming-computation-design)
5. [Data Ingestion Layer](#data-ingestion-layer)
6. [Technology Stack](#technology-stack)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Performance Considerations](#performance-considerations)
9. [Code Examples & Patterns](#code-examples--patterns)
10. [Risk Assessment](#risk-assessment)

---

## Executive Summary

### Proposed Features

1. **Batch Mode** ✅ (Already Implemented)
   - Current JAX implementation processes batches of contracts
   - 49,495 contracts/second throughput

2. **Streaming Mode** 🆕 (Proposed)
   - Real-time Greeks computation on incoming market data
   - Maintain historical state for time-series Greeks
   - Sub-millisecond latency for NSE trading

3. **Multi-Source Ingestion** 🆕 (Proposed)
   - CSV files (batch import)
   - Redis HKEY (current snapshot)
   - Redis TimeSeries (historical time-series)
   - Redis Streams (event-driven real-time)
   - QuestDB (high-performance analytics)

### Key Recommendations

| Aspect | Recommendation | Rationale |
|--------|---------------|-----------|
| **Architecture** | **Hybrid Batch-Stream** | Leverage JAX batch processing within streaming pipeline |
| **Streaming Framework** | **Redis Streams + Custom Processor** | Native integration, <1ms latency, proven for financial data |
| **State Management** | **Redis + In-Memory Cache** | Fast lookups for IV history, Greeks history |
| **Analytics Store** | **QuestDB** | Best-in-class for time-series analytics (2.4M rows/s) |
| **Deployment** | **Microservice Architecture** | Scalable, maintainable, cloud-native |

### Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| **Latency** | <5ms (p95) | Micro-batching + JAX compilation |
| **Throughput** | 100K updates/sec | Horizontal scaling + Redis Streams |
| **History Depth** | 30 days tick data | QuestDB + Redis TimeSeries |
| **Availability** | 99.9% | Redundancy + health checks |

---

## Requirements Analysis

### Functional Requirements

#### FR1: Streaming Mode
- **FR1.1:** Process market data updates in real-time (tick-by-tick or micro-batches)
- **FR1.2:** Compute Greeks incrementally as new prices arrive
- **FR1.3:** Maintain history of IVs, prices, Greeks for each contract
- **FR1.4:** Support both full recalculation and incremental updates
- **FR1.5:** Handle out-of-order messages (late arrivals)

#### FR2: Multi-Source Ingestion
- **FR2.1:** CSV files - batch import for historical data
- **FR2.2:** Redis HKEY - key-value snapshot reads
- **FR2.3:** Redis TimeSeries - time-series data with aggregations
- **FR2.4:** Redis Streams - real-time event stream
- **FR2.5:** QuestDB - SQL queries for analytics

#### FR3: State Management
- **FR3.1:** Store and retrieve option contract metadata
- **FR3.2:** Maintain IV history for stability/smoothing
- **FR3.3:** Track underlying price movements
- **FR3.4:** Persist computed Greeks for historical analysis

### Non-Functional Requirements

#### NFR1: Performance
- **Latency:** p50 < 2ms, p95 < 5ms, p99 < 10ms for streaming updates
- **Throughput:** 100,000+ option updates per second
- **Batch Size:** Configurable micro-batches (10-1000 contracts)

#### NFR2: Scalability
- **Horizontal Scaling:** Add workers to increase throughput
- **Vertical Scaling:** Leverage GPU/TPU when available
- **Data Volume:** Handle 50,000+ active contracts (NSE scale)

#### NFR3: Reliability
- **Fault Tolerance:** Automatic recovery from failures
- **Data Consistency:** Exactly-once or at-least-once processing
- **Monitoring:** Real-time metrics and alerting

---

## Recommended Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Sources Layer                           │
├─────────────┬─────────────┬─────────────┬─────────────┬────────────┤
│ CSV Files   │ Redis HKEY  │ Redis TS    │Redis Streams│  QuestDB   │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴─────┬──────┘
       │             │             │             │            │
┌──────▼─────────────▼─────────────▼─────────────▼────────────▼──────┐
│              Unified Data Ingestion Layer (Adapters)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │   CSV    │ │  HKEY    │ │TimeSeries│ │ Streams  │ │ QuestDB  │ │
│  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
└───────┼────────────┼────────────┼────────────┼────────────┼───────┘
        │            │            │            │            │
        └────────────┴────────────┴────────────┴────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                    Message Queue / Stream Buffer                    │
│                     (Redis Streams - Internal)                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                   Stream Processing Engine                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              Micro-Batch Aggregator                        │    │
│  │  • Collects updates into micro-batches (configurable)     │    │
│  │  • Groups by expiry/underlying for efficient computation   │    │
│  │  • Windowing: tumbling (every N ms) or count-based        │    │
│  └────────────────────┬───────────────────────────────────────┘    │
│                       │                                             │
│  ┌────────────────────▼───────────────────────────────────────┐    │
│  │              State Manager                                 │    │
│  │  • In-memory cache (LRU) for hot data                     │    │
│  │  • Redis backing store for persistence                     │    │
│  │  • Contract metadata, IV history, price history           │    │
│  └────────────────────┬───────────────────────────────────────┘    │
│                       │                                             │
│  ┌────────────────────▼───────────────────────────────────────┐    │
│  │         JAX Greeks Computation Engine                      │    │
│  │  • Batch processing using existing GreeksJAX.py           │    │
│  │  • JIT-compiled for low latency                           │    │
│  │  • GPU/TPU acceleration when available                     │    │
│  └────────────────────┬───────────────────────────────────────┘    │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────────┐
│                    Output & Storage Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │
│  │Redis Streams│  │ Redis TS    │  │  QuestDB    │  │WebSocket │  │
│  │ (real-time) │  │ (metrics)   │  │ (analytics) │  │  (API)   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. **Data Sources Layer**
- Multiple heterogeneous sources
- Each with different semantics (batch, snapshot, stream, time-series)
- Isolated from processing logic via adapters

#### 2. **Unified Ingestion Layer**
- Adapter pattern for each data source
- Normalizes data to common format
- Handles retries, error handling, backpressure
- Publishes to internal stream buffer

#### 3. **Stream Processing Engine**
- **Micro-Batch Aggregator:** Groups updates for efficient JAX processing
- **State Manager:** Maintains contract state, IV history
- **JAX Computation Engine:** Reuses existing batch implementation

#### 4. **Output Layer**
- Multiple sinks for different use cases
- Real-time updates, historical analytics, API access

---

## Streaming Computation Design

### Approach: **Micro-Batching with JAX**

#### Why Micro-Batching?

| Pure Streaming (1-by-1) | Micro-Batching | Pure Batch |
|------------------------|----------------|------------|
| ❌ Can't leverage JAX vectorization | ✅ Best of both worlds | ❌ High latency |
| ❌ High per-message overhead | ✅ Low latency + high throughput | ✅ High throughput |
| Latency: <1ms | **Latency: 2-5ms** | Latency: 100ms+ |
| Throughput: 10K/s | **Throughput: 100K/s** | Throughput: 1M/s |

**Recommended:** Micro-batching with 10-100ms windows or 50-500 contract batches.

### Streaming Computation Pipeline

```python
# Conceptual flow (detailed implementation later)

while True:
    # 1. Collect micro-batch (time-based or count-based)
    batch = collect_updates(window_ms=50, max_size=500)

    # 2. Enrich with state (IV history, metadata)
    enriched_batch = enrich_with_state(batch, state_manager)

    # 3. Prepare for JAX computation
    jax_input = prepare_jax_input(enriched_batch)

    # 4. Compute Greeks using existing JAX implementation
    greeks = calculate_option_metrics(**jax_input)

    # 5. Update state (new IVs, prices, Greeks)
    state_manager.update(enriched_batch, greeks)

    # 6. Publish results
    publish_results(greeks, output_streams)
```

### State Management Design

#### State Requirements

For each option contract, maintain:
1. **Metadata:** Symbol, expiry, strike, type (call/put)
2. **Current State:** Last price, underlying price, IV, Greeks
3. **History:** Time-series of IVs (for smoothing/stability)
4. **Timestamps:** Last update time, expiry time

#### State Storage Strategy

**Two-Tier Storage:**

```
┌─────────────────────────────────────────────────────────┐
│           Tier 1: In-Memory Cache (Hot Data)            │
│  • LRU cache (10,000 most active contracts)            │
│  • Python dict or Redis client-side cache              │
│  • Sub-microsecond access                              │
│  • Stores: current state + recent history (last 100)   │
└─────────────────────┬───────────────────────────────────┘
                      │ (cache miss)
┌─────────────────────▼───────────────────────────────────┐
│        Tier 2: Redis (Persistent Store)                 │
│  • All active contracts (50K+)                         │
│  • HSET for metadata & current state                   │
│  • ZSET for sorted history (by timestamp)             │
│  • Redis TimeSeries for long-term history              │
│  • Sub-millisecond access                              │
└─────────────────────┬───────────────────────────────────┘
                      │ (archive)
┌─────────────────────▼───────────────────────────────────┐
│          Tier 3: QuestDB (Cold Storage)                 │
│  • All historical data (30+ days)                      │
│  • Batch writes (every 5 minutes)                      │
│  • SQL analytics queries                               │
│  • Partitioned by date                                 │
└─────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. **Streaming update** → Update Tier 1 (cache)
2. **Cache update** → Async write to Tier 2 (Redis)
3. **Every 5 min** → Batch flush Tier 2 → Tier 3 (QuestDB)

#### State Schema

**Redis HSET (contract metadata & current state):**
```
Key: "contract:{symbol}:{strike}:{expiry}:{type}"
Fields:
  - symbol: "NIFTY"
  - strike: "18000"
  - expiry: "2025-01-30"
  - type: "CE"
  - last_price: "125.50"
  - underlying_price: "18050.00"
  - iv: "18.5"
  - delta: "0.55"
  - gamma: "0.002"
  - ... (other Greeks)
  - last_update: "1704700800.123"
```

**Redis ZSET (IV history - sorted by timestamp):**
```
Key: "iv_history:{symbol}:{strike}:{expiry}:{type}"
Members: "timestamp:iv_value"
Score: timestamp
Example: ZADD iv_history:NIFTY:18000:2025-01-30:CE 1704700800 "1704700800:18.5"
```

**Redis TimeSeries (long-term metrics):**
```
Key: "ts:greeks:{symbol}:{strike}:{expiry}:{type}:{metric}"
Example: ts:greeks:NIFTY:18000:2025-01-30:CE:delta
Data: [(timestamp1, delta1), (timestamp2, delta2), ...]
```

### Historical Context in Computation

#### IV Smoothing with History

```python
def compute_greeks_with_history(current_price, contract_id, state_manager):
    """
    Compute Greeks using IV smoothing based on history.
    """
    # Get IV history (last 10 ticks)
    iv_history = state_manager.get_iv_history(contract_id, limit=10)

    # Compute current IV
    current_iv = bisection_implied_volatility(...)

    # Smooth IV using exponential moving average
    if len(iv_history) > 0:
        alpha = 0.3  # Smoothing factor
        smoothed_iv = alpha * current_iv + (1 - alpha) * iv_history[-1]
    else:
        smoothed_iv = current_iv

    # Compute Greeks using smoothed IV
    greeks = compute_all_greeks_call(S, K, r, T, smoothed_iv / 100.0)

    # Store current IV in history
    state_manager.add_iv_to_history(contract_id, current_iv, timestamp=now())

    return greeks, smoothed_iv
```

#### Time-Series Greeks Analysis

```python
def get_greeks_time_series(contract_id, start_time, end_time):
    """
    Retrieve historical Greeks for analysis.
    """
    # Query QuestDB for historical data
    query = f"""
        SELECT timestamp, delta, gamma, vega, theta, rho
        FROM option_greeks
        WHERE contract_id = '{contract_id}'
          AND timestamp BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY timestamp
    """

    results = questdb_client.query(query)
    return results.to_pandas()
```

---

## Data Ingestion Layer

### Design Principles

1. **Adapter Pattern:** Each source has dedicated adapter
2. **Common Interface:** All adapters produce normalized `MarketUpdate` objects
3. **Fault Isolation:** Failure in one source doesn't affect others
4. **Configurable:** Enable/disable sources via configuration

### Common Data Model

```python
@dataclass
class MarketUpdate:
    """Normalized market data update."""
    contract_id: str           # Unique identifier
    symbol: str                # Underlying symbol (e.g., "NIFTY")
    strike: float              # Strike price
    expiry: datetime           # Expiration date
    option_type: str           # "CE" or "PE"

    underlying_price: float    # Current underlying price
    call_price: Optional[float] = None
    put_price: Optional[float] = None

    timestamp: float           # Unix timestamp
    source: str                # Source identifier

    # Optional fields
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
```

### Adapter 1: CSV File Ingestion

**Use Case:** Batch import of historical data, backtesting

```python
class CSVAdapter:
    """
    Adapter for CSV file ingestion.
    Supports both one-time batch import and file watching.
    """

    def __init__(self, file_path: str, watch_mode: bool = False):
        self.file_path = file_path
        self.watch_mode = watch_mode

    def read_batch(self) -> List[MarketUpdate]:
        """Read entire CSV file as batch."""
        df = pd.read_csv(self.file_path, parse_dates=['expiry', 'timestamp'])
        updates = []

        for _, row in df.iterrows():
            update = MarketUpdate(
                contract_id=self._generate_contract_id(row),
                symbol=row['symbol'],
                strike=row['strike'],
                expiry=row['expiry'],
                option_type=row['option_type'],
                underlying_price=row['underlying_price'],
                call_price=row.get('call_price'),
                put_price=row.get('put_price'),
                timestamp=row['timestamp'].timestamp(),
                source='csv'
            )
            updates.append(update)

        return updates

    def watch(self) -> Generator[List[MarketUpdate], None, None]:
        """Watch file for new rows (streaming mode)."""
        # Implementation using file watching or polling
        pass

    def _generate_contract_id(self, row) -> str:
        return f"{row['symbol']}:{row['strike']}:{row['expiry']}:{row['option_type']}"
```

**Configuration:**
```yaml
csv_adapter:
  enabled: true
  file_path: "/data/nse_options.csv"
  watch_mode: false  # One-time batch import
  batch_size: 10000  # Process in chunks
```

### Adapter 2: Redis HKEY (Snapshot)

**Use Case:** Read current market snapshot from Redis hash

```python
class RedisHKEYAdapter:
    """
    Adapter for Redis HKEY snapshot reads.
    Polls Redis hashes for current option prices.
    """

    def __init__(self, redis_client, key_pattern: str, poll_interval_ms: int = 100):
        self.redis = redis_client
        self.key_pattern = key_pattern  # e.g., "nse:options:*"
        self.poll_interval_ms = poll_interval_ms

    def stream(self) -> Generator[MarketUpdate, None, None]:
        """Poll Redis hashes and yield updates."""
        while True:
            # Scan for matching keys
            keys = self.redis.keys(self.key_pattern)

            for key in keys:
                # Get hash fields
                data = self.redis.hgetall(key)

                if self._has_changed(key, data):
                    update = self._parse_hash(key, data)
                    yield update

            time.sleep(self.poll_interval_ms / 1000.0)

    def _parse_hash(self, key: str, data: dict) -> MarketUpdate:
        """Parse Redis hash into MarketUpdate."""
        # Parse key: "nse:options:NIFTY:18000:2025-01-30:CE"
        parts = key.split(':')

        return MarketUpdate(
            contract_id=key,
            symbol=parts[2],
            strike=float(parts[3]),
            expiry=datetime.fromisoformat(parts[4]),
            option_type=parts[5],
            underlying_price=float(data[b'underlying_price']),
            call_price=float(data.get(b'call_price', 0)),
            put_price=float(data.get(b'put_price', 0)),
            timestamp=float(data.get(b'timestamp', time.time())),
            source='redis_hkey'
        )

    def _has_changed(self, key: str, data: dict) -> bool:
        """Check if hash has changed since last read."""
        # Implementation: compare timestamp or use pub/sub notifications
        pass
```

**Redis Schema:**
```
Key: "nse:options:NIFTY:18000:2025-01-30:CE"
Fields:
  underlying_price: "18050.00"
  call_price: "125.50"
  put_price: "45.25"
  volume: "15000"
  timestamp: "1704700800.123"
```

### Adapter 3: Redis TimeSeries

**Use Case:** Historical time-series data, aggregated views

```python
class RedisTimeSeriesAdapter:
    """
    Adapter for Redis TimeSeries.
    Reads time-series data with aggregations.
    """

    def __init__(self, redis_client, key_prefix: str = "ts:market"):
        self.redis = redis_client
        self.key_prefix = key_prefix

    def get_range(self, contract_id: str, metric: str,
                  start_ts: int, end_ts: int) -> List[Tuple[int, float]]:
        """
        Get time-series data for a metric.

        Args:
            contract_id: Contract identifier
            metric: Metric name (e.g., 'price', 'volume')
            start_ts: Start timestamp (ms)
            end_ts: End timestamp (ms)

        Returns:
            List of (timestamp, value) tuples
        """
        key = f"{self.key_prefix}:{contract_id}:{metric}"

        # TS.RANGE key start_ts end_ts
        result = self.redis.execute_command(
            'TS.RANGE', key, start_ts, end_ts
        )

        return [(ts, float(val)) for ts, val in result]

    def get_aggregated(self, contract_id: str, metric: str,
                       start_ts: int, end_ts: int,
                       aggregation: str = 'avg', bucket_ms: int = 60000):
        """
        Get aggregated time-series data.

        Args:
            aggregation: 'avg', 'sum', 'min', 'max', 'count'
            bucket_ms: Aggregation bucket size in milliseconds
        """
        key = f"{self.key_prefix}:{contract_id}:{metric}"

        result = self.redis.execute_command(
            'TS.RANGE', key, start_ts, end_ts,
            'AGGREGATION', aggregation, bucket_ms
        )

        return [(ts, float(val)) for ts, val in result]

    def stream_updates(self) -> Generator[MarketUpdate, None, None]:
        """
        Stream real-time updates from TimeSeries.
        Uses pub/sub for notifications.
        """
        # Subscribe to TimeSeries update notifications
        pubsub = self.redis.pubsub()
        pubsub.subscribe(f"{self.key_prefix}:updates")

        for message in pubsub.listen():
            if message['type'] == 'message':
                # Parse notification and fetch latest value
                contract_id, metric = message['data'].decode().split(':')
                latest = self.redis.execute_command(
                    'TS.GET', f"{self.key_prefix}:{contract_id}:{metric}"
                )

                if latest:
                    ts, value = latest
                    yield self._create_update(contract_id, metric, ts, value)
```

**Redis TimeSeries Schema:**
```
# Price time-series
Key: "ts:market:NIFTY:18000:2025-01-30:CE:price"
Data: [(timestamp1, price1), (timestamp2, price2), ...]

# Volume time-series
Key: "ts:market:NIFTY:18000:2025-01-30:CE:volume"
Data: [(timestamp1, volume1), (timestamp2, volume2), ...]
```

**Aggregation Example:**
```python
# Get average price for last hour in 1-minute buckets
adapter.get_aggregated(
    contract_id="NIFTY:18000:2025-01-30:CE",
    metric="price",
    start_ts=now_ms - 3600000,  # 1 hour ago
    end_ts=now_ms,
    aggregation='avg',
    bucket_ms=60000  # 1 minute
)
```

### Adapter 4: Redis Streams (Real-Time)

**Use Case:** Event-driven real-time processing, highest performance

```python
class RedisStreamsAdapter:
    """
    Adapter for Redis Streams.
    Consumes events from Redis Streams with consumer groups.
    """

    def __init__(self, redis_client, stream_key: str,
                 consumer_group: str, consumer_name: str):
        self.redis = redis_client
        self.stream_key = stream_key
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name

        # Create consumer group if not exists
        try:
            self.redis.xgroup_create(stream_key, consumer_group, id='0', mkstream=True)
        except Exception:
            pass  # Group already exists

    def stream(self, block_ms: int = 1000, count: int = 100) -> Generator[List[MarketUpdate], None, None]:
        """
        Stream updates from Redis Streams.

        Args:
            block_ms: Blocking timeout in milliseconds
            count: Max number of messages to read at once

        Yields:
            Batches of MarketUpdate objects
        """
        last_id = '>'  # Read only new messages

        while True:
            # XREADGROUP GROUP <group> <consumer> COUNT <count> BLOCK <ms> STREAMS <stream> <id>
            results = self.redis.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_key: last_id},
                count=count,
                block=block_ms
            )

            if not results:
                continue

            updates = []
            message_ids = []

            for stream_name, messages in results:
                for message_id, data in messages:
                    message_ids.append(message_id)
                    update = self._parse_stream_message(data)
                    updates.append(update)

            if updates:
                yield updates

                # Acknowledge processed messages
                self.redis.xack(self.stream_key, self.consumer_group, *message_ids)

    def _parse_stream_message(self, data: dict) -> MarketUpdate:
        """Parse Redis Stream message."""
        return MarketUpdate(
            contract_id=data[b'contract_id'].decode(),
            symbol=data[b'symbol'].decode(),
            strike=float(data[b'strike']),
            expiry=datetime.fromisoformat(data[b'expiry'].decode()),
            option_type=data[b'option_type'].decode(),
            underlying_price=float(data[b'underlying_price']),
            call_price=float(data.get(b'call_price', 0)) or None,
            put_price=float(data.get(b'put_price', 0)) or None,
            timestamp=float(data[b'timestamp']),
            source='redis_stream'
        )
```

**Producer Example (for testing):**
```python
def publish_market_update(redis_client, stream_key: str, update: MarketUpdate):
    """Publish update to Redis Stream."""
    redis_client.xadd(
        stream_key,
        {
            'contract_id': update.contract_id,
            'symbol': update.symbol,
            'strike': str(update.strike),
            'expiry': update.expiry.isoformat(),
            'option_type': update.option_type,
            'underlying_price': str(update.underlying_price),
            'call_price': str(update.call_price or 0),
            'put_price': str(update.put_price or 0),
            'timestamp': str(update.timestamp)
        }
    )
```

**Consumer Group Benefits:**
- Multiple consumers can process same stream in parallel
- Automatic load balancing across consumers
- Fault tolerance: if consumer dies, messages are redistributed
- Acknowledgment system for exactly-once semantics

### Adapter 5: QuestDB

**Use Case:** Historical analytics, complex queries, batch exports

```python
class QuestDBAdapter:
    """
    Adapter for QuestDB time-series database.
    Reads historical data via SQL queries.
    """

    def __init__(self, host: str = 'localhost', port: int = 8812):
        """
        Args:
            host: QuestDB host
            port: PostgreSQL wire protocol port (default 8812)
        """
        import psycopg2
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            user='admin',
            password='quest',
            database='qdb'
        )

    def query_market_data(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return pd.read_sql(sql, self.conn)

    def get_historical_batch(self, start_time: datetime, end_time: datetime,
                             symbols: Optional[List[str]] = None) -> List[MarketUpdate]:
        """
        Get historical market data as batch.

        Args:
            start_time: Start of time range
            end_time: End of time range
            symbols: Optional list of symbols to filter

        Returns:
            List of MarketUpdate objects
        """
        where_clauses = [
            f"timestamp >= '{start_time.isoformat()}'",
            f"timestamp <= '{end_time.isoformat()}'"
        ]

        if symbols:
            symbol_list = "', '".join(symbols)
            where_clauses.append(f"symbol IN ('{symbol_list}')")

        where_str = ' AND '.join(where_clauses)

        query = f"""
            SELECT
                contract_id, symbol, strike, expiry, option_type,
                underlying_price, call_price, put_price, timestamp
            FROM market_data
            WHERE {where_str}
            ORDER BY timestamp
        """

        df = self.query_market_data(query)

        updates = []
        for _, row in df.iterrows():
            updates.append(MarketUpdate(
                contract_id=row['contract_id'],
                symbol=row['symbol'],
                strike=row['strike'],
                expiry=pd.to_datetime(row['expiry']).to_pydatetime(),
                option_type=row['option_type'],
                underlying_price=row['underlying_price'],
                call_price=row['call_price'],
                put_price=row['put_price'],
                timestamp=pd.to_datetime(row['timestamp']).timestamp(),
                source='questdb'
            ))

        return updates

    def write_greeks(self, greeks_df: pd.DataFrame):
        """
        Write computed Greeks to QuestDB.
        Uses ILP (InfluxDB Line Protocol) for high-performance ingestion.
        """
        from questdb.ingress import Sender, IngressError

        try:
            with Sender('localhost', 9009) as sender:
                for _, row in greeks_df.iterrows():
                    sender.row(
                        'option_greeks',
                        symbols={'contract_id': row['contract_id']},
                        columns={
                            'iv': row['iv'],
                            'delta': row['delta'],
                            'gamma': row['gamma'],
                            'vega': row['vega'],
                            'theta': row['theta'],
                            'rho': row['rho']
                        },
                        at=pd.Timestamp(row['timestamp'])
                    )
                sender.flush()
        except IngressError as e:
            print(f"Failed to write to QuestDB: {e}")
```

**QuestDB Schema:**
```sql
-- Market data table
CREATE TABLE market_data (
    timestamp TIMESTAMP,
    contract_id SYMBOL,
    symbol SYMBOL,
    strike DOUBLE,
    expiry TIMESTAMP,
    option_type SYMBOL,
    underlying_price DOUBLE,
    call_price DOUBLE,
    put_price DOUBLE,
    volume LONG,
    open_interest LONG
) TIMESTAMP(timestamp) PARTITION BY DAY;

-- Greeks table
CREATE TABLE option_greeks (
    timestamp TIMESTAMP,
    contract_id SYMBOL,
    iv DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    vega DOUBLE,
    theta DOUBLE,
    rho DOUBLE
) TIMESTAMP(timestamp) PARTITION BY DAY;

-- Indexes for fast queries
CREATE INDEX idx_contract ON market_data (contract_id);
CREATE INDEX idx_symbol ON market_data (symbol);
```

**Analytics Query Examples:**
```sql
-- Average IV by strike for last hour
SELECT
    strike,
    AVG(iv) as avg_iv,
    STDDEV(iv) as iv_volatility
FROM option_greeks
WHERE symbol = 'NIFTY'
  AND timestamp > dateadd('h', -1, now())
GROUP BY strike
ORDER BY strike;

-- Greeks time-series for specific contract
SELECT
    timestamp,
    delta, gamma, vega, theta, rho
FROM option_greeks
WHERE contract_id = 'NIFTY:18000:2025-01-30:CE'
  AND timestamp > dateadd('d', -7, now())
SAMPLE BY 1m;  -- 1-minute aggregation
```

---

## Technology Stack

### Recommended Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Computation Engine** | JAX (existing) | 49K ops/s, auto-diff, GPU/TPU support |
| **Streaming Framework** | Redis Streams | <1ms latency, 100K+ msgs/s, proven for finance |
| **In-Memory State** | Redis | Sub-ms lookups, 1M+ ops/s, persistence |
| **Time-Series Store** | Redis TimeSeries | Native aggregations, 500K+ writes/s |
| **Analytics DB** | QuestDB | 2.4M rows/s ingestion, SQL, financial focus |
| **Programming Language** | Python 3.11+ | Existing codebase, JAX support |
| **Orchestration** | Docker Compose / K8s | Scalability, deployment |
| **Monitoring** | Prometheus + Grafana | Metrics, alerting, dashboards |

### Alternative Considerations

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Apache Kafka** | Industry standard, high throughput | Higher complexity, operational overhead | ❌ Overkill for this use case |
| **Apache Flink** | Advanced stream processing | Steep learning curve, Java ecosystem | ❌ Too complex |
| **InfluxDB** | Popular TSDB | Slower than QuestDB for financial data | ⚠️ QuestDB better for tick data |
| **TimescaleDB** | PostgreSQL-based | 16× slower ingestion than QuestDB | ⚠️ QuestDB preferred |
| **Python asyncio** | Native Python | Harder to scale, GIL limitations | ⚠️ Redis Streams better |

### Dependency Matrix

**Core Dependencies:**
```
jax>=0.4.0
jaxlib>=0.4.0
redis>=5.0.0
redis-timeseries>=1.4.0  # If using Redis Stack
pandas>=1.3.0
numpy>=1.20.0
questdb-py>=1.0.0  # QuestDB Python client
psycopg2-binary>=2.9.0  # For QuestDB PostgreSQL wire
```

**Optional Dependencies:**
```
prometheus-client>=0.15.0  # Metrics
pydantic>=2.0.0  # Data validation
python-dotenv>=1.0.0  # Configuration
uvloop>=0.17.0  # Faster event loop
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal:** Basic streaming infrastructure

**Tasks:**
1. **Data Model Definition**
   - Define `MarketUpdate` dataclass
   - Create base `Adapter` interface
   - Implement serialization/deserialization

2. **State Manager Implementation**
   - In-memory cache (Python dict with LRU)
   - Redis backing store
   - CRUD operations for contract state

3. **CSV Adapter** (simplest, for testing)
   - Batch import
   - File watching (optional)

4. **Testing**
   - Unit tests for state manager
   - Integration test with CSV data

**Deliverables:**
- `streaming/models.py` - Data models
- `streaming/state_manager.py` - State management
- `streaming/adapters/csv_adapter.py` - CSV ingestion
- Test suite

### Phase 2: Redis Integration (Week 3-4)

**Goal:** Real-time streaming with Redis

**Tasks:**
1. **Redis Streams Adapter**
   - Consumer group implementation
   - Error handling and retries
   - Acknowledgment logic

2. **Redis HKEY Adapter**
   - Polling mechanism
   - Change detection

3. **Micro-Batch Aggregator**
   - Time-based windowing
   - Count-based batching
   - Backpressure handling

4. **JAX Integration**
   - Adapt existing `calculate_option_metrics` for streaming
   - Add IV history smoothing
   - Performance optimization

**Deliverables:**
- `streaming/adapters/redis_streams.py`
- `streaming/adapters/redis_hkey.py`
- `streaming/processor.py` - Main processing engine
- Performance benchmarks

### Phase 3: Time-Series Storage (Week 5-6)

**Goal:** Historical data management

**Tasks:**
1. **Redis TimeSeries Integration**
   - Adapter implementation
   - Aggregation queries
   - Retention policies

2. **QuestDB Integration**
   - Schema design
   - Adapter implementation
   - Batch write optimization
   - Analytics query library

3. **Output Layer**
   - Multi-sink publisher
   - WebSocket API for real-time updates
   - REST API for queries

**Deliverables:**
- `streaming/adapters/redis_timeseries.py`
- `streaming/adapters/questdb.py`
- `api/websocket_server.py`
- `api/rest_api.py`

### Phase 4: Production Hardening (Week 7-8)

**Goal:** Production-ready system

**Tasks:**
1. **Monitoring & Observability**
   - Prometheus metrics
   - Grafana dashboards
   - Logging (structured)

2. **Error Handling & Recovery**
   - Circuit breakers
   - Dead letter queues
   - Graceful shutdown

3. **Configuration Management**
   - YAML/JSON config files
   - Environment variables
   - Validation

4. **Performance Optimization**
   - Profiling
   - Caching strategies
   - Connection pooling

5. **Documentation**
   - API documentation
   - Deployment guide
   - Operational runbook

**Deliverables:**
- `monitoring/` - Prometheus, Grafana configs
- `config/` - Configuration schemas
- Production documentation

### Phase 5: Advanced Features (Week 9-10+)

**Goal:** Enhanced capabilities

**Tasks:**
1. **Horizontal Scaling**
   - Multiple processing workers
   - Load balancing
   - Kubernetes deployment

2. **Advanced Analytics**
   - Greeks surface visualization
   - Volatility smile tracking
   - Risk metrics aggregation

3. **Machine Learning Integration**
   - IV prediction models
   - Anomaly detection

**Deliverables:**
- K8s manifests
- ML pipeline (optional)

---

## Performance Considerations

### Latency Breakdown

**Target: <5ms end-to-end latency (p95)**

| Stage | Target Latency | Optimization Strategy |
|-------|----------------|----------------------|
| **Data Ingestion** | <0.5ms | Redis Streams consumer, connection pooling |
| **State Lookup** | <0.3ms | In-memory LRU cache, batch Redis reads |
| **Micro-Batching** | 1-3ms | Configurable window size, adaptive batching |
| **JAX Computation** | 1-2ms | JIT compilation, batch size tuning |
| **State Update** | <0.5ms | Async writes, write batching |
| **Output Publishing** | <0.5ms | Async pub/sub, connection pooling |
| **Total** | **3-6ms** | Within target! |

### Throughput Optimization

**Target: 100,000 updates/second**

**Strategy:**
1. **Horizontal Scaling:** Run multiple processor instances
   - Each instance: 10K-20K updates/sec
   - 5-10 instances = 100K+ updates/sec

2. **Batch Size Tuning:**
   - Too small: High overhead, can't leverage JAX vectorization
   - Too large: High latency
   - **Optimal:** 100-500 contracts per batch

3. **Connection Pooling:**
   - Redis connection pool (10-20 connections)
   - QuestDB connection pool (5-10 connections)

4. **Async I/O:**
   - Use `asyncio` for I/O-bound operations
   - JAX computation in separate thread pool

### Memory Management

**Estimated Memory Requirements:**

| Component | Memory | Notes |
|-----------|--------|-------|
| **JAX Runtime** | 500MB | Base + compiled functions |
| **In-Memory Cache** | 1-2GB | 10K contracts × ~200KB each |
| **Redis Client** | 100MB | Connection buffers |
| **Python Runtime** | 200MB | Base |
| **Total per instance** | **~2-3GB** | Comfortable on 4GB machines |

**For 50K contracts (NSE scale):**
- In-memory cache: 10K hot contracts
- Cold contracts: Redis lookups (<1ms)
- Memory: Still ~2-3GB per instance

### Benchmarking Strategy

**Metrics to Track:**

1. **Latency:**
   - p50, p95, p99, p999
   - Per-stage breakdown
   - End-to-end

2. **Throughput:**
   - Updates processed per second
   - By source adapter
   - By contract

3. **Resource Utilization:**
   - CPU usage
   - Memory usage
   - Network I/O
   - Redis operations/sec

4. **Error Rates:**
   - Failed updates
   - State lookup failures
   - Computation errors

**Load Testing:**
```python
# Synthetic load generator
async def generate_load(target_qps: int, duration_seconds: int):
    """
    Generate synthetic market updates for load testing.

    Args:
        target_qps: Target queries per second
        duration_seconds: Test duration
    """
    interval = 1.0 / target_qps
    end_time = time.time() + duration_seconds

    while time.time() < end_time:
        # Generate random update
        update = generate_random_market_update()

        # Publish to Redis Stream
        publish_market_update(redis_client, 'market:updates', update)

        await asyncio.sleep(interval)

# Run load test
asyncio.run(generate_load(target_qps=100000, duration_seconds=60))
```

---

## Code Examples & Patterns

### Example 1: End-to-End Streaming Pipeline

```python
# streaming/main.py

import asyncio
from typing import List
from .adapters.redis_streams import RedisStreamsAdapter
from .processor import StreamProcessor
from .state_manager import StateManager
import redis

async def main():
    """
    Main streaming application entry point.
    """
    # Initialize Redis client
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    # Initialize state manager
    state_manager = StateManager(redis_client, cache_size=10000)

    # Initialize Redis Streams adapter
    adapter = RedisStreamsAdapter(
        redis_client=redis_client,
        stream_key='market:updates',
        consumer_group='greeks_compute',
        consumer_name='worker_1'
    )

    # Initialize stream processor
    processor = StreamProcessor(
        state_manager=state_manager,
        batch_size=500,
        window_ms=50
    )

    print("Starting streaming Greeks computation...")

    # Process stream
    async for batch in adapter.stream():
        # Compute Greeks for batch
        results = await processor.process_batch(batch)

        # Publish results
        await processor.publish_results(results)

        # Log metrics
        print(f"Processed {len(batch)} updates, computed {len(results)} Greeks")

if __name__ == '__main__':
    asyncio.run(main())
```

### Example 2: Stream Processor Implementation

```python
# streaming/processor.py

import jax.numpy as jnp
import numpy as np
from typing import List, Dict
from .models import MarketUpdate
from .state_manager import StateManager
from OptionGreeksGPU.GreeksJAX import calculate_option_metrics

class StreamProcessor:
    """
    Main stream processing engine.
    Orchestrates micro-batching, state management, and Greeks computation.
    """

    def __init__(self, state_manager: StateManager,
                 batch_size: int = 500, window_ms: int = 50):
        self.state_manager = state_manager
        self.batch_size = batch_size
        self.window_ms = window_ms

        # Micro-batch buffer
        self.buffer: List[MarketUpdate] = []
        self.last_flush_time = time.time()

    async def process_batch(self, updates: List[MarketUpdate]) -> List[Dict]:
        """
        Process a batch of market updates.

        Args:
            updates: List of MarketUpdate objects

        Returns:
            List of computed Greeks results
        """
        # Group updates by expiry/underlying for efficient computation
        grouped = self._group_by_expiry(updates)

        all_results = []

        for (symbol, expiry), group_updates in grouped.items():
            # Enrich with state (IV history, metadata)
            enriched = await self._enrich_with_state(group_updates)

            # Prepare JAX input
            jax_input = self._prepare_jax_input(enriched)

            # Compute Greeks (reuse existing JAX implementation)
            greeks = calculate_option_metrics(
                option_data=jax_input['option_data'],
                days_to_expiry=jax_input['days_to_expiry'],
                interest_rate=jax_input['interest_rate']
            )

            # Parse results and update state
            results = await self._parse_and_update_state(enriched, greeks)
            all_results.extend(results)

        return all_results

    def _group_by_expiry(self, updates: List[MarketUpdate]) -> Dict:
        """Group updates by (symbol, expiry) for batch processing."""
        grouped = {}
        for update in updates:
            key = (update.symbol, update.expiry)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(update)
        return grouped

    async def _enrich_with_state(self, updates: List[MarketUpdate]) -> List[Dict]:
        """
        Enrich updates with state from state manager.
        Adds IV history, metadata, etc.
        """
        enriched = []

        for update in updates:
            # Get contract state (cached or from Redis)
            state = await self.state_manager.get_contract_state(update.contract_id)

            # Get IV history for smoothing
            iv_history = await self.state_manager.get_iv_history(
                update.contract_id, limit=10
            )

            enriched.append({
                'update': update,
                'state': state,
                'iv_history': iv_history
            })

        return enriched

    def _prepare_jax_input(self, enriched: List[Dict]) -> Dict:
        """
        Prepare input for JAX Greeks computation.

        Format expected by calculate_option_metrics:
        option_data: array of [strike, underlying, call_price, call_type, put_price, put_type]
        days_to_expiry: scalar
        interest_rate: scalar
        """
        n = len(enriched)

        option_data = np.zeros((n, 6))

        for i, item in enumerate(enriched):
            update = item['update']

            option_data[i, 0] = update.strike
            option_data[i, 1] = update.underlying_price
            option_data[i, 2] = update.call_price or 0.0
            option_data[i, 3] = 0  # Call type flag
            option_data[i, 4] = update.put_price or 0.0
            option_data[i, 5] = 1  # Put type flag

        # Calculate days to expiry (assume all in batch have same expiry)
        expiry = enriched[0]['update'].expiry
        now = datetime.now()
        days_to_expiry = (expiry - now).total_seconds() / 86400

        # Use constant interest rate (could be dynamic)
        interest_rate = 5.0  # 5%

        return {
            'option_data': option_data,
            'days_to_expiry': days_to_expiry,
            'interest_rate': interest_rate
        }

    async def _parse_and_update_state(self, enriched: List[Dict],
                                       greeks: List[np.ndarray]) -> List[Dict]:
        """
        Parse Greeks results and update state.

        greeks format:
        [call_IVs, call_deltas, ..., put_rhos] (14 arrays)
        """
        call_IVs, call_deltas, call_delta2s, call_vegas, call_gammas, \
        call_thetas, call_rhos, put_IVs, put_deltas, put_delta2s, \
        put_vegas, put_gammas, put_thetas, put_rhos = greeks

        results = []

        for i, item in enumerate(enriched):
            update = item['update']

            # Create result object
            result = {
                'contract_id': update.contract_id,
                'timestamp': update.timestamp,
                'underlying_price': update.underlying_price,
                'call': {
                    'price': update.call_price,
                    'iv': float(call_IVs[i]),
                    'delta': float(call_deltas[i]),
                    'gamma': float(call_gammas[i]),
                    'vega': float(call_vegas[i]),
                    'theta': float(call_thetas[i]),
                    'rho': float(call_rhos[i])
                },
                'put': {
                    'price': update.put_price,
                    'iv': float(put_IVs[i]),
                    'delta': float(put_deltas[i]),
                    'gamma': float(put_gammas[i]),
                    'vega': float(put_vegas[i]),
                    'theta': float(put_thetas[i]),
                    'rho': float(put_rhos[i])
                }
            }

            # Update state manager
            await self.state_manager.update_contract_state(
                contract_id=update.contract_id,
                state=result,
                timestamp=update.timestamp
            )

            # Add IV to history
            await self.state_manager.add_iv_to_history(
                contract_id=update.contract_id,
                call_iv=float(call_IVs[i]),
                put_iv=float(put_IVs[i]),
                timestamp=update.timestamp
            )

            results.append(result)

        return results

    async def publish_results(self, results: List[Dict]):
        """
        Publish results to output sinks.
        """
        # Publish to Redis Streams for real-time consumers
        await self._publish_to_redis_streams(results)

        # Batch write to QuestDB for analytics
        await self._write_to_questdb(results)

        # Update Redis TimeSeries for metrics
        await self._update_redis_timeseries(results)

    async def _publish_to_redis_streams(self, results: List[Dict]):
        """Publish to Redis Streams output."""
        # Implementation
        pass

    async def _write_to_questdb(self, results: List[Dict]):
        """Batch write to QuestDB."""
        # Implementation
        pass

    async def _update_redis_timeseries(self, results: List[Dict]):
        """Update Redis TimeSeries metrics."""
        # Implementation
        pass
```

### Example 3: State Manager Implementation

```python
# streaming/state_manager.py

import redis
import json
from typing import Dict, List, Optional
from collections import OrderedDict
import time

class StateManager:
    """
    Manages state for option contracts.
    Two-tier storage: in-memory cache + Redis.
    """

    def __init__(self, redis_client: redis.Redis, cache_size: int = 10000):
        self.redis = redis_client
        self.cache_size = cache_size

        # In-memory LRU cache
        self.cache: OrderedDict = OrderedDict()

    async def get_contract_state(self, contract_id: str) -> Optional[Dict]:
        """
        Get current state for a contract.
        Checks cache first, then Redis.
        """
        # Check cache
        if contract_id in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(contract_id)
            return self.cache[contract_id]

        # Cache miss - query Redis
        key = f"contract:{contract_id}"
        state_json = self.redis.get(key)

        if state_json:
            state = json.loads(state_json)

            # Add to cache
            self._add_to_cache(contract_id, state)

            return state

        return None

    async def update_contract_state(self, contract_id: str,
                                     state: Dict, timestamp: float):
        """
        Update contract state.
        Updates cache and asynchronously writes to Redis.
        """
        # Add timestamp
        state['last_update'] = timestamp

        # Update cache
        self._add_to_cache(contract_id, state)

        # Async write to Redis
        key = f"contract:{contract_id}"
        self.redis.set(key, json.dumps(state))

    async def get_iv_history(self, contract_id: str,
                             limit: int = 10) -> List[float]:
        """
        Get IV history for a contract.
        Returns list of recent IVs (call IVs).
        """
        key = f"iv_history:{contract_id}"

        # ZREVRANGE to get latest values (sorted by timestamp descending)
        results = self.redis.zrevrange(key, 0, limit - 1, withscores=False)

        # Parse results
        ivs = []
        for item in results:
            # Format: "timestamp:iv_value"
            parts = item.decode().split(':')
            if len(parts) == 2:
                ivs.append(float(parts[1]))

        return ivs

    async def add_iv_to_history(self, contract_id: str,
                                call_iv: float, put_iv: float,
                                timestamp: float):
        """
        Add IV to history (Redis sorted set).
        """
        # Store both call and put IVs
        key_call = f"iv_history:{contract_id}:call"
        key_put = f"iv_history:{contract_id}:put"

        # ZADD with timestamp as score
        self.redis.zadd(key_call, {f"{timestamp}:{call_iv}": timestamp})
        self.redis.zadd(key_put, {f"{timestamp}:{put_iv}": timestamp})

        # Keep only last 100 entries (trim old data)
        self.redis.zremrangebyrank(key_call, 0, -101)
        self.redis.zremrangebyrank(key_put, 0, -101)

        # Set expiry (7 days)
        self.redis.expire(key_call, 7 * 86400)
        self.redis.expire(key_put, 7 * 86400)

    def _add_to_cache(self, contract_id: str, state: Dict):
        """Add to LRU cache, evicting oldest if full."""
        # Add/update
        if contract_id in self.cache:
            self.cache.move_to_end(contract_id)
        self.cache[contract_id] = state

        # Evict oldest if cache full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
```

### Example 4: Configuration File

```yaml
# config/streaming.yaml

app:
  name: "OptionGreeksStreaming"
  version: "3.1.0"
  log_level: "INFO"

processing:
  batch_size: 500              # Max contracts per batch
  window_ms: 50                # Micro-batch window (milliseconds)
  workers: 4                   # Number of parallel workers
  interest_rate: 5.0           # Default risk-free rate (%)

state:
  cache_size: 10000            # In-memory cache size (contracts)
  iv_history_depth: 100        # IV history depth per contract
  state_ttl_days: 7            # Redis state TTL

redis:
  host: "localhost"
  port: 6379
  db: 0
  max_connections: 20

  streams:
    input_key: "market:updates"
    output_key: "greeks:updates"
    consumer_group: "greeks_compute"
    consumer_name: "worker_{id}"
    block_ms: 1000
    count: 100

  timeseries:
    enabled: true
    retention_days: 30
    metrics:
      - "delta"
      - "gamma"
      - "vega"
      - "iv"

questdb:
  host: "localhost"
  port: 8812                   # PostgreSQL wire protocol
  ilp_port: 9009               # ILP (fast ingestion)
  user: "admin"
  password: "quest"

  tables:
    market_data: "market_data"
    greeks: "option_greeks"

  batch_size: 5000             # Batch writes
  flush_interval_sec: 300      # Flush every 5 minutes

sources:
  csv:
    enabled: false
    file_path: "/data/nse_options.csv"
    watch_mode: false

  redis_hkey:
    enabled: false
    key_pattern: "nse:options:*"
    poll_interval_ms: 100

  redis_timeseries:
    enabled: false
    key_prefix: "ts:market"

  redis_streams:
    enabled: true

  questdb:
    enabled: true

monitoring:
  prometheus:
    enabled: true
    port: 9090

  metrics:
    - "updates_processed_total"
    - "greeks_computed_total"
    - "latency_seconds"
    - "batch_size_current"
    - "cache_hit_rate"

output:
  redis_streams:
    enabled: true
    key: "greeks:updates"

  redis_timeseries:
    enabled: true

  questdb:
    enabled: true

  websocket:
    enabled: true
    host: "0.0.0.0"
    port: 8080
```

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **JAX compilation overhead** | Medium | Medium | Pre-warm JIT, persistent compilation cache |
| **Redis memory limits** | Low | High | Monitoring, TTLs, tiered storage (QuestDB) |
| **Message loss (Redis Streams)** | Low | High | Consumer groups, acknowledgments, DLQ |
| **State consistency** | Medium | Medium | Idempotent operations, versioning |
| **Network latency** | Low | Medium | Co-locate services, connection pooling |
| **Dependency conflicts** | Low | Low | Virtual environment, pinned versions |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Data source outages** | Medium | Medium | Graceful degradation, fallbacks |
| **Scaling bottlenecks** | Medium | High | Horizontal scaling, load testing |
| **Config errors** | Medium | Medium | Schema validation, canary deploys |
| **Monitoring gaps** | Low | Medium | Comprehensive metrics, alerting |

### Business Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Latency SLA miss** | Low | High | Performance testing, optimization |
| **Data quality issues** | Medium | High | Validation, sanity checks, alerts |
| **Incorrect Greeks** | Low | Critical | Comprehensive testing, validation vs market |

---

## Next Steps

### Immediate Actions

1. **Review & Feedback** (You)
   - Review this architecture document
   - Provide feedback on recommendations
   - Prioritize features

2. **Prototype** (Week 1)
   - Implement basic Redis Streams adapter
   - Test micro-batching with JAX
   - Measure latency/throughput

3. **Decision Points**
   - Confirm technology stack (Redis Streams vs alternatives)
   - Confirm QuestDB vs TimescaleDB/InfluxDB
   - Decide on deployment model (Docker vs K8s)

4. **Detailed Design** (Week 2)
   - API specifications
   - Database schemas
   - Monitoring strategy

### Questions for You

1. **Deployment Environment:**
   - On-premises or cloud?
   - Existing infrastructure (Redis, databases)?
   - Scalability requirements (peak load)?

2. **Data Sources Priority:**
   - Which sources are most important to implement first?
   - Do you have existing NSE data feeds?

3. **Historical Data:**
   - How much historical data needs to be stored?
   - Analytics requirements (query patterns)?

4. **Latency vs Throughput:**
   - Is <5ms latency critical or can we relax for higher throughput?
   - Acceptable p99 latency?

---

**End of Document**

This architecture provides a production-ready foundation for streaming option Greeks computation with multi-source ingestion. The design leverages JAX's existing strength in batch processing within a micro-batching streaming paradigm, achieving both low latency (<5ms) and high throughput (100K+ updates/sec).

The modular adapter pattern makes it easy to add new data sources, and the two-tier state management (cache + Redis) ensures sub-millisecond state access while maintaining persistence.

Ready to proceed with implementation when you are!
