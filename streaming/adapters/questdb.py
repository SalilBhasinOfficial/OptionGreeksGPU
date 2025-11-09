"""QuestDB adapter for SQL analytics and batch exports."""

import pandas as pd
from typing import List, Generator, Optional
from datetime import datetime

from .base import BaseAdapter
from ..models import MarketUpdate, OptionType, DataSource


class QuestDBAdapter(BaseAdapter):
    """Adapter for QuestDB time-series database."""

    def __init__(self, host: str = 'localhost', port: int = 8812):
        super().__init__("QuestDBAdapter")
        self.host = host
        self.port = port
        self.conn = None

    def connect(self) -> bool:
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user='admin',
                password='quest',
                database='qdb'
            )
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to QuestDB: {e}")
            return False

    def disconnect(self):
        if self.conn:
            self.conn.close()
        self.is_connected = False

    def read_batch(self, limit: Optional[int] = None) -> List[MarketUpdate]:
        """Read batch from QuestDB."""
        if not self.is_connected:
            self.connect()

        query = "SELECT * FROM market_data ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"

        try:
            df = pd.read_sql(query, self.conn)
            updates = []
            for _, row in df.iterrows():
                # Parse and create MarketUpdate
                updates.append(self._parse_row(row))
            return [u for u in updates if u]
        except:
            return []

    def stream(self) -> Generator[List[MarketUpdate], None, None]:
        # QuestDB is primarily for analytics, not streaming
        while True:
            yield []

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query."""
        if not self.is_connected:
            self.connect()
        return pd.read_sql(sql, self.conn)

    def write_greeks(self, greeks_data: List[dict]):
        """Write Greeks to QuestDB via ILP protocol."""
        try:
            from questdb.ingress import Sender
            with Sender(self.host, 9009) as sender:
                for data in greeks_data:
                    sender.row(
                        'option_greeks',
                        symbols={'contract_id': data['contract_id']},
                        columns={k: v for k, v in data.items() if k != 'contract_id'},
                        at=pd.Timestamp(data.get('timestamp', datetime.now()))
                    )
                sender.flush()
        except:
            pass

    def _parse_row(self, row: pd.Series) -> Optional[MarketUpdate]:
        """Parse row to MarketUpdate."""
        try:
            return MarketUpdate(
                contract_id=row['contract_id'],
                symbol=row['symbol'],
                strike=float(row['strike']),
                expiry=pd.to_datetime(row['expiry']).to_pydatetime(),
                option_type=OptionType(row['option_type']),
                underlying_price=float(row['underlying_price']),
                call_price=float(row.get('call_price')) if pd.notna(row.get('call_price')) else None,
                put_price=float(row.get('put_price')) if pd.notna(row.get('put_price')) else None,
                timestamp=pd.to_datetime(row['timestamp']).timestamp(),
                source=DataSource.QUESTDB
            )
        except:
            return None
