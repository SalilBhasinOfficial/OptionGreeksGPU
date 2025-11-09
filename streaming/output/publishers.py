"""
Multi-sink publisher for option Greeks results.

Publishes to multiple destinations:
- Redis Streams (real-time consumers)
- Redis TimeSeries (metrics)
- QuestDB (analytics)
"""

import json
import time
from typing import List, Dict, Any
from datetime import datetime

from ..models import GreeksResult


class MultiSinkPublisher:
    """
    Publisher that sends results to multiple sinks.
    """

    def __init__(self, redis_client=None, questdb_adapter=None,
                 output_stream_key: str = "greeks:updates",
                 enable_redis_streams: bool = True,
                 enable_redis_timeseries: bool = True,
                 enable_questdb: bool = True):
        """
        Initialize multi-sink publisher.

        Args:
            redis_client: Redis client instance
            questdb_adapter: QuestDB adapter instance
            output_stream_key: Redis Stream key for output
            enable_redis_streams: Enable Redis Streams output
            enable_redis_timeseries: Enable Redis TimeSeries output
            enable_questdb: Enable QuestDB output
        """
        self.redis = redis_client
        self.questdb = questdb_adapter
        self.output_stream_key = output_stream_key
        self.enable_redis_streams = enable_redis_streams and redis_client is not None
        self.enable_redis_timeseries = enable_redis_timeseries and redis_client is not None
        self.enable_questdb = enable_questdb and questdb_adapter is not None

        # Buffer for batch writes
        self.questdb_buffer = []
        self.questdb_buffer_size = 1000
        self.last_questdb_flush = time.time()
        self.questdb_flush_interval = 300  # 5 minutes

        # Statistics
        self.stats = {
            'total_published': 0,
            'redis_streams_published': 0,
            'redis_timeseries_published': 0,
            'questdb_published': 0,
            'publish_errors': 0
        }

    def publish(self, results: List[GreeksResult]) -> int:
        """
        Publish results to all enabled sinks.

        Args:
            results: List of GreeksResult objects

        Returns:
            Number of results successfully published
        """
        if not results:
            return 0

        published_count = 0

        try:
            # Publish to Redis Streams (real-time)
            if self.enable_redis_streams:
                self._publish_to_redis_streams(results)
                self.stats['redis_streams_published'] += len(results)

            # Publish to Redis TimeSeries (metrics)
            if self.enable_redis_timeseries:
                self._publish_to_redis_timeseries(results)
                self.stats['redis_timeseries_published'] += len(results)

            # Buffer for QuestDB (analytics)
            if self.enable_questdb:
                self._buffer_for_questdb(results)

            published_count = len(results)
            self.stats['total_published'] += published_count

        except Exception as e:
            print(f"Error publishing results: {e}")
            self.stats['publish_errors'] += 1

        return published_count

    def _publish_to_redis_streams(self, results: List[GreeksResult]):
        """Publish to Redis Streams for real-time consumers."""
        if not self.redis:
            return

        for result in results:
            try:
                data = {
                    'contract_id': result.contract_id,
                    'timestamp': str(result.timestamp),
                    'underlying_price': str(result.underlying_price),
                    'call_iv': str(result.call.get('iv', 0)),
                    'call_delta': str(result.call.get('delta', 0)),
                    'call_gamma': str(result.call.get('gamma', 0)),
                    'call_vega': str(result.call.get('vega', 0)),
                    'call_theta': str(result.call.get('theta', 0)),
                    'call_rho': str(result.call.get('rho', 0)),
                    'put_iv': str(result.put.get('iv', 0)),
                    'put_delta': str(result.put.get('delta', 0)),
                    'put_gamma': str(result.put.get('gamma', 0)),
                    'put_vega': str(result.put.get('vega', 0)),
                    'put_theta': str(result.put.get('theta', 0)),
                    'put_rho': str(result.put.get('rho', 0))
                }

                self.redis.xadd(self.output_stream_key, data)

            except Exception as e:
                print(f"Error publishing to Redis Streams: {e}")

    def _publish_to_redis_timeseries(self, results: List[GreeksResult]):
        """Publish to Redis TimeSeries for metrics."""
        if not self.redis:
            return

        for result in results:
            try:
                ts = int(result.timestamp * 1000)  # Milliseconds

                # Publish key metrics
                for metric_name, value in [
                    ('call_iv', result.call.get('iv', 0)),
                    ('call_delta', result.call.get('delta', 0)),
                    ('put_iv', result.put.get('iv', 0)),
                    ('put_delta', result.put.get('delta', 0))
                ]:
                    key = f"ts:greeks:{result.contract_id}:{metric_name}"
                    try:
                        self.redis.execute_command('TS.ADD', key, ts, value)
                    except:
                        # Key might not exist, create it
                        try:
                            self.redis.execute_command('TS.CREATE', key,
                                                     'RETENTION', '2592000000')  # 30 days
                            self.redis.execute_command('TS.ADD', key, ts, value)
                        except:
                            pass

            except Exception as e:
                print(f"Error publishing to Redis TimeSeries: {e}")

    def _buffer_for_questdb(self, results: List[GreeksResult]):
        """Buffer results for QuestDB batch write."""
        for result in results:
            self.questdb_buffer.append({
                'contract_id': result.contract_id,
                'timestamp': result.timestamp,
                'underlying_price': result.underlying_price,
                **{f'call_{k}': v for k, v in result.call.items()},
                **{f'put_{k}': v for k, v in result.put.items()}
            })

        # Flush if buffer full or interval exceeded
        if (len(self.questdb_buffer) >= self.questdb_buffer_size or
            time.time() - self.last_questdb_flush >= self.questdb_flush_interval):
            self.flush_questdb()

    def flush_questdb(self):
        """Flush buffered results to QuestDB."""
        if not self.questdb or not self.questdb_buffer:
            return

        try:
            self.questdb.write_greeks(self.questdb_buffer)
            self.stats['questdb_published'] += len(self.questdb_buffer)
            self.questdb_buffer.clear()
            self.last_questdb_flush = time.time()

        except Exception as e:
            print(f"Error flushing to QuestDB: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        return {
            **self.stats,
            'questdb_buffer_size': len(self.questdb_buffer)
        }

    def close(self):
        """Close publisher and flush remaining data."""
        if self.questdb_buffer:
            self.flush_questdb()
