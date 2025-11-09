"""
Base adapter interface for data ingestion.

All data source adapters must implement this interface to ensure
consistent behavior across different sources.
"""

from abc import ABC, abstractmethod
from typing import List, Generator, Optional
from ..models import MarketUpdate


class BaseAdapter(ABC):
    """
    Abstract base class for all data source adapters.

    Adapters are responsible for:
    1. Connecting to a data source
    2. Normalizing data to MarketUpdate format
    3. Handling errors and retries
    4. Providing consistent interface for streaming/batch reads
    """

    def __init__(self, name: str):
        """
        Initialize adapter.

        Args:
            name: Adapter name for logging/identification
        """
        self.name = name
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to data source.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from data source."""
        pass

    @abstractmethod
    def read_batch(self, limit: Optional[int] = None) -> List[MarketUpdate]:
        """
        Read a batch of market updates.

        Args:
            limit: Maximum number of updates to read (None = all)

        Returns:
            List of MarketUpdate objects
        """
        pass

    @abstractmethod
    def stream(self) -> Generator[List[MarketUpdate], None, None]:
        """
        Stream market updates continuously.

        Yields:
            Batches of MarketUpdate objects
        """
        pass

    def validate_update(self, update: MarketUpdate) -> bool:
        """
        Validate a market update.

        Args:
            update: MarketUpdate to validate

        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        if update.strike <= 0:
            return False
        if update.underlying_price <= 0:
            return False
        if update.call_price is not None and update.call_price < 0:
            return False
        if update.put_price is not None and update.put_price < 0:
            return False

        return True

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
