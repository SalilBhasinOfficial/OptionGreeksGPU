"""
Data models for streaming option Greeks computation.

This module defines the core data structures used throughout the streaming pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "CE"
    PUT = "PE"


class DataSource(Enum):
    """Data source enumeration."""
    CSV = "csv"
    REDIS_HKEY = "redis_hkey"
    REDIS_TIMESERIES = "redis_timeseries"
    REDIS_STREAMS = "redis_streams"
    QUESTDB = "questdb"


@dataclass
class MarketUpdate:
    """
    Normalized market data update from any source.

    This is the common data model that all adapters produce.
    """
    # Contract identification
    contract_id: str           # Unique identifier (e.g., "NIFTY:18000:2025-01-30:CE")
    symbol: str                # Underlying symbol (e.g., "NIFTY")
    strike: float              # Strike price
    expiry: datetime           # Expiration date
    option_type: OptionType    # CALL or PUT

    # Market data
    underlying_price: float    # Current underlying price
    call_price: Optional[float] = None   # Call option market price
    put_price: Optional[float] = None    # Put option market price

    # Metadata
    timestamp: float = 0.0     # Unix timestamp
    source: DataSource = DataSource.CSV

    # Optional fields
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'strike': self.strike,
            'expiry': self.expiry.isoformat(),
            'option_type': self.option_type.value,
            'underlying_price': self.underlying_price,
            'call_price': self.call_price,
            'put_price': self.put_price,
            'timestamp': self.timestamp,
            'source': self.source.value,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'bid': self.bid,
            'ask': self.ask
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketUpdate':
        """Create from dictionary."""
        return cls(
            contract_id=data['contract_id'],
            symbol=data['symbol'],
            strike=float(data['strike']),
            expiry=datetime.fromisoformat(data['expiry']),
            option_type=OptionType(data['option_type']),
            underlying_price=float(data['underlying_price']),
            call_price=float(data['call_price']) if data.get('call_price') else None,
            put_price=float(data['put_price']) if data.get('put_price') else None,
            timestamp=float(data.get('timestamp', 0)),
            source=DataSource(data.get('source', 'csv')),
            volume=int(data['volume']) if data.get('volume') else None,
            open_interest=int(data['open_interest']) if data.get('open_interest') else None,
            bid=float(data['bid']) if data.get('bid') else None,
            ask=float(data['ask']) if data.get('ask') else None
        )


@dataclass
class ContractState:
    """
    Current state of an option contract.

    Stored in StateManager for historical context.
    """
    contract_id: str
    symbol: str
    strike: float
    expiry: datetime
    option_type: OptionType

    # Current market data
    underlying_price: float
    call_price: Optional[float] = None
    put_price: Optional[float] = None

    # Computed Greeks (call)
    call_iv: Optional[float] = None
    call_delta: Optional[float] = None
    call_gamma: Optional[float] = None
    call_vega: Optional[float] = None
    call_theta: Optional[float] = None
    call_rho: Optional[float] = None
    call_delta2: Optional[float] = None

    # Computed Greeks (put)
    put_iv: Optional[float] = None
    put_delta: Optional[float] = None
    put_gamma: Optional[float] = None
    put_vega: Optional[float] = None
    put_theta: Optional[float] = None
    put_rho: Optional[float] = None
    put_delta2: Optional[float] = None

    # Metadata
    last_update: float = 0.0
    update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            'contract_id': self.contract_id,
            'symbol': self.symbol,
            'strike': self.strike,
            'expiry': self.expiry.isoformat(),
            'option_type': self.option_type.value,
            'underlying_price': self.underlying_price,
            'call_price': self.call_price,
            'put_price': self.put_price,
            'call_iv': self.call_iv,
            'call_delta': self.call_delta,
            'call_gamma': self.call_gamma,
            'call_vega': self.call_vega,
            'call_theta': self.call_theta,
            'call_rho': self.call_rho,
            'call_delta2': self.call_delta2,
            'put_iv': self.put_iv,
            'put_delta': self.put_delta,
            'put_gamma': self.put_gamma,
            'put_vega': self.put_vega,
            'put_theta': self.put_theta,
            'put_rho': self.put_rho,
            'put_delta2': self.put_delta2,
            'last_update': self.last_update,
            'update_count': self.update_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContractState':
        """Create from dictionary."""
        return cls(
            contract_id=data['contract_id'],
            symbol=data['symbol'],
            strike=float(data['strike']),
            expiry=datetime.fromisoformat(data['expiry']),
            option_type=OptionType(data['option_type']),
            underlying_price=float(data['underlying_price']),
            call_price=float(data['call_price']) if data.get('call_price') else None,
            put_price=float(data['put_price']) if data.get('put_price') else None,
            call_iv=float(data['call_iv']) if data.get('call_iv') else None,
            call_delta=float(data['call_delta']) if data.get('call_delta') else None,
            call_gamma=float(data['call_gamma']) if data.get('call_gamma') else None,
            call_vega=float(data['call_vega']) if data.get('call_vega') else None,
            call_theta=float(data['call_theta']) if data.get('call_theta') else None,
            call_rho=float(data['call_rho']) if data.get('call_rho') else None,
            call_delta2=float(data['call_delta2']) if data.get('call_delta2') else None,
            put_iv=float(data['put_iv']) if data.get('put_iv') else None,
            put_delta=float(data['put_delta']) if data.get('put_delta') else None,
            put_gamma=float(data['put_gamma']) if data.get('put_gamma') else None,
            put_vega=float(data['put_vega']) if data.get('put_vega') else None,
            put_theta=float(data['put_theta']) if data.get('put_theta') else None,
            put_rho=float(data['put_rho']) if data.get('put_rho') else None,
            put_delta2=float(data['put_delta2']) if data.get('put_delta2') else None,
            last_update=float(data.get('last_update', 0)),
            update_count=int(data.get('update_count', 0))
        )


@dataclass
class GreeksResult:
    """
    Result of Greeks computation for a single contract.
    """
    contract_id: str
    timestamp: float
    underlying_price: float

    # Call Greeks
    call: Dict[str, float] = field(default_factory=dict)

    # Put Greeks
    put: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'contract_id': self.contract_id,
            'timestamp': self.timestamp,
            'underlying_price': self.underlying_price,
            'call': self.call,
            'put': self.put
        }


@dataclass
class ProcessingMetrics:
    """
    Metrics for monitoring streaming performance.
    """
    batch_size: int = 0
    processing_time_ms: float = 0.0
    jax_compute_time_ms: float = 0.0
    state_lookup_time_ms: float = 0.0
    state_update_time_ms: float = 0.0
    output_publish_time_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'processing_time_ms': self.processing_time_ms,
            'jax_compute_time_ms': self.jax_compute_time_ms,
            'state_lookup_time_ms': self.state_lookup_time_ms,
            'state_update_time_ms': self.state_update_time_ms,
            'output_publish_time_ms': self.output_publish_time_ms,
            'timestamp': self.timestamp
        }
