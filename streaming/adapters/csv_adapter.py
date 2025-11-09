"""
CSV file adapter for batch import and file watching.

Supports:
- One-time batch import
- Continuous file watching for new rows
- Standard NSE CSV format
"""

import pandas as pd
import time
from typing import List, Generator, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .base import BaseAdapter
from ..models import MarketUpdate, OptionType, DataSource


class CSVAdapter(BaseAdapter):
    """
    Adapter for CSV file ingestion.

    Supports both batch import and file watching mode.
    """

    def __init__(self, file_path: str, watch_mode: bool = False,
                 batch_size: int = 1000, poll_interval_sec: float = 1.0):
        """
        Initialize CSV adapter.

        Args:
            file_path: Path to CSV file
            watch_mode: If True, watch file for new rows
            batch_size: Number of rows to process per batch
            poll_interval_sec: Polling interval for watch mode (seconds)
        """
        super().__init__("CSVAdapter")
        self.file_path = Path(file_path)
        self.watch_mode = watch_mode
        self.batch_size = batch_size
        self.poll_interval_sec = poll_interval_sec

        # For watch mode
        self.last_read_position = 0
        self.last_file_size = 0

    def connect(self) -> bool:
        """
        Connect to CSV file (verify it exists).

        Returns:
            True if file exists
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.is_connected = True
        return True

    def disconnect(self):
        """Disconnect (no-op for CSV)."""
        self.is_connected = False

    def read_batch(self, limit: Optional[int] = None) -> List[MarketUpdate]:
        """
        Read entire CSV file as batch.

        Args:
            limit: Maximum number of updates to read (None = all)

        Returns:
            List of MarketUpdate objects
        """
        if not self.is_connected:
            self.connect()

        # Read CSV
        df = pd.read_csv(self.file_path)

        # Parse dates if columns exist
        if 'expiry' in df.columns:
            df['expiry'] = pd.to_datetime(df['expiry'])
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Use current time as timestamp
            df['timestamp'] = pd.Timestamp.now()

        # Limit if specified
        if limit:
            df = df.head(limit)

        # Convert to MarketUpdate objects
        updates = []
        for _, row in df.iterrows():
            update = self._parse_row(row)
            if update and self.validate_update(update):
                updates.append(update)

        return updates

    def stream(self) -> Generator[List[MarketUpdate], None, None]:
        """
        Stream updates from CSV file.

        If watch_mode=False: Yields entire file in batches, then stops
        If watch_mode=True: Continuously watches file for new rows

        Yields:
            Batches of MarketUpdate objects
        """
        if not self.is_connected:
            self.connect()

        if not self.watch_mode:
            # One-time batch read
            all_updates = self.read_batch()

            # Yield in batches
            for i in range(0, len(all_updates), self.batch_size):
                batch = all_updates[i:i + self.batch_size]
                yield batch
        else:
            # Watch mode - continuously monitor file
            while True:
                new_updates = self._read_new_rows()
                if new_updates:
                    # Yield in batches
                    for i in range(0, len(new_updates), self.batch_size):
                        batch = new_updates[i:i + self.batch_size]
                        yield batch

                # Sleep before next poll
                time.sleep(self.poll_interval_sec)

    def _read_new_rows(self) -> List[MarketUpdate]:
        """
        Read only new rows appended to file (watch mode).

        Returns:
            List of new MarketUpdate objects
        """
        current_size = self.file_path.stat().st_size

        # Check if file has grown
        if current_size <= self.last_file_size:
            return []

        # Read from last position
        try:
            df = pd.read_csv(self.file_path, skiprows=self.last_read_position)

            # Update position
            self.last_read_position += len(df)
            self.last_file_size = current_size

            # Parse dates
            if 'expiry' in df.columns:
                df['expiry'] = pd.to_datetime(df['expiry'])
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = pd.Timestamp.now()

            # Convert to MarketUpdate objects
            updates = []
            for _, row in df.iterrows():
                update = self._parse_row(row)
                if update and self.validate_update(update):
                    updates.append(update)

            return updates

        except Exception as e:
            print(f"Error reading new rows: {e}")
            return []

    def _parse_row(self, row: pd.Series) -> Optional[MarketUpdate]:
        """
        Parse a DataFrame row into MarketUpdate.

        Expected columns:
        - symbol: Underlying symbol
        - strike: Strike price
        - expiry: Expiration date
        - option_type: "CE" or "PE" (or "CALL"/"PUT")
        - underlying_price: Current underlying price
        - call_price: Call option price (optional)
        - put_price: Put option price (optional)
        - timestamp: Update timestamp (optional)

        Args:
            row: pandas Series

        Returns:
            MarketUpdate or None if parsing fails
        """
        try:
            # Parse option type
            option_type_str = str(row.get('option_type', 'CE')).upper()
            if option_type_str in ['CE', 'CALL']:
                option_type = OptionType.CALL
            elif option_type_str in ['PE', 'PUT']:
                option_type = OptionType.PUT
            else:
                return None

            # Generate contract ID
            symbol = str(row['symbol'])
            strike = float(row['strike'])
            expiry = row['expiry']

            if isinstance(expiry, str):
                expiry = pd.to_datetime(expiry).to_pydatetime()
            elif isinstance(expiry, pd.Timestamp):
                expiry = expiry.to_pydatetime()

            contract_id = f"{symbol}:{strike}:{expiry.strftime('%Y-%m-%d')}:{option_type.value}"

            # Get timestamp
            if 'timestamp' in row and pd.notna(row['timestamp']):
                ts = row['timestamp']
                if isinstance(ts, str):
                    timestamp = pd.to_datetime(ts).timestamp()
                elif isinstance(ts, pd.Timestamp):
                    timestamp = ts.timestamp()
                else:
                    timestamp = time.time()
            else:
                timestamp = time.time()

            # Create MarketUpdate
            update = MarketUpdate(
                contract_id=contract_id,
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                underlying_price=float(row['underlying_price']),
                call_price=float(row['call_price']) if 'call_price' in row and pd.notna(row['call_price']) else None,
                put_price=float(row['put_price']) if 'put_price' in row and pd.notna(row['put_price']) else None,
                timestamp=timestamp,
                source=DataSource.CSV,
                volume=int(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None,
                open_interest=int(row['open_interest']) if 'open_interest' in row and pd.notna(row['open_interest']) else None,
                bid=float(row['bid']) if 'bid' in row and pd.notna(row['bid']) else None,
                ask=float(row['ask']) if 'ask' in row and pd.notna(row['ask']) else None
            )

            return update

        except Exception as e:
            print(f"Error parsing row: {e}")
            return None


def create_sample_csv(file_path: str, num_contracts: int = 100):
    """
    Create a sample CSV file for testing.

    Args:
        file_path: Path to create CSV file
        num_contracts: Number of sample contracts
    """
    import numpy as np

    data = []
    symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']

    # Use a future expiry date (30 days from now)
    expiry_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

    for i in range(num_contracts):
        symbol = np.random.choice(symbols)

        if symbol == 'NIFTY':
            underlying = 18000 + np.random.randn() * 100
            strike = round(underlying / 50) * 50
        elif symbol == 'BANKNIFTY':
            underlying = 42000 + np.random.randn() * 200
            strike = round(underlying / 100) * 100
        else:
            underlying = 19000 + np.random.randn() * 100
            strike = round(underlying / 50) * 50

        option_type = np.random.choice(['CE', 'PE'])

        # Simplified pricing (not realistic, just for testing)
        if option_type == 'CE':
            call_price = max(0, underlying - strike + np.random.randn() * 20)
            put_price = max(0, strike - underlying + np.random.randn() * 20)
        else:
            call_price = max(0, underlying - strike + np.random.randn() * 20)
            put_price = max(0, strike - underlying + np.random.randn() * 20)

        data.append({
            'symbol': symbol,
            'strike': strike,
            'expiry': expiry_date,
            'option_type': option_type,
            'underlying_price': underlying,
            'call_price': call_price,
            'put_price': put_price,
            'volume': np.random.randint(1000, 50000),
            'open_interest': np.random.randint(10000, 500000)
        })

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Created sample CSV with {num_contracts} contracts at {file_path}")
