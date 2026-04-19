import os
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class OLAPStore(ABC):
    @abstractmethod
    def write_applications(self, df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def write_transactions(self, df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def read_applications(self) -> pl.DataFrame:
        pass

    @abstractmethod
    def read_transactions(self) -> pl.DataFrame:
        pass

    @abstractmethod
    def get_last_processed_id(self) -> int:
        pass

    @abstractmethod
    def set_last_processed_id(self, app_id: int) -> None:
        pass


class DuckDBStore(OLAPStore):
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.environ.get("OLAP_DB_PATH", "data/olap.duckdb")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS app_id_seq START 1
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                application_id INTEGER PRIMARY KEY,
                applicant_name VARCHAR,
                email VARCHAR,
                ssn_last4 VARCHAR,
                annual_income DECIMAL,
                requested_amount DECIMAL,
                employment_status VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY,
                application_id INTEGER REFERENCES applications(application_id),
                amount DECIMAL,
                merchant VARCHAR,
                category VARCHAR,
                transaction_time TIMESTAMP,
                location_country VARCHAR,
                is_online BOOLEAN
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_state (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)
        conn.close()

    def write_applications(self, df: pl.DataFrame) -> None:
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        conn.execute("DELETE FROM applications")
        conn.execute("""
            INSERT INTO applications
            SELECT
                application_id,
                applicant_name,
                email,
                ssn_last4,
                annual_income,
                requested_amount,
                employment_status,
                status,
                created_at
            FROM df
        """)
        conn.close()

    def write_transactions(self, df: pl.DataFrame) -> None:
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        conn.execute("DELETE FROM transactions")
        conn.execute("""
            INSERT INTO transactions
            SELECT
                id,
                application_id,
                amount,
                merchant,
                category,
                transaction_time,
                location_country,
                is_online
            FROM df
        """)
        conn.close()

    def read_applications(self) -> pl.DataFrame:
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        df = conn.execute("SELECT * FROM applications ORDER BY application_id").pl()
        conn.close()
        return df

    def read_transactions(self) -> pl.DataFrame:
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        df = conn.execute("SELECT * FROM transactions ORDER BY application_id, transaction_time").pl()
        conn.close()
        return df

    def get_last_processed_id(self) -> int:
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        result = conn.execute("SELECT value FROM pipeline_state WHERE key = 'last_processed_id'").fetchone()
        conn.close()
        if result:
            return int(result[0])
        return 0

    def set_last_processed_id(self, app_id: int) -> None:
        import duckdb

        conn = duckdb.connect(str(self.db_path))
        conn.execute("""
            INSERT OR REPLACE INTO pipeline_state (key, value)
            VALUES ('last_processed_id', CAST(? AS VARCHAR))
        """, (str(app_id),))
        conn.close()


class DatabricksStore(OLAPStore):
    def __init__(
        self,
        server_hostname: str | None = None,
        http_path: str | None = None,
        catalog: str | None = None,
    ):
        self.server_hostname = server_hostname or os.environ.get("DATABRICKS_SERVER_HOSTNAME")
        self.http_path = http_path or os.environ.get("DATABRICKS_HTTP_PATH")
        self.access_token = os.environ.get("DATABRICKS_ACCESS_TOKEN")
        self.catalog = catalog or os.environ.get("DATABRICKS_CATALOG", "hive_metastore")
        self._conn = None
        self._connected = False

    def _ensure_connection(self):
        if self._conn is not None:
            return
        if not self.server_hostname or not self.http_path:
            raise ValueError(
                "Databricks credentials not configured. Set:\n"
                "  DATABRICKS_SERVER_HOSTNAME\n"
                "  DATABRICKS_HTTP_PATH\n"
                "  DATABRICKS_ACCESS_TOKEN"
            )
        import duckdb

        conn_str = (
            f"databricks://token:{self.access_token}@"
            f"{self.server_hostname}?http_path={self.http_path}&catalog={self.catalog}"
        )
        self._conn = duckdb.connect(conn_str)
        self._connected = True

    def write_applications(self, df: pl.DataFrame) -> None:
        self._ensure_connection()
        import duckdb

        temp = duckdb.connect(":memory:")
        temp.execute("CREATE TABLE applications AS SELECT * FROM df")
        temp.execute(
            f"INSERT INTO '{self.catalog}.fraud.applications' SELECT * FROM applications"
        )
        temp.close()

    def write_transactions(self, df: pl.DataFrame) -> None:
        self._ensure_connection()
        import duckdb

        temp = duckdb.connect(":memory:")
        temp.execute("CREATE TABLE transactions AS SELECT * FROM df")
        temp.execute(
            f"INSERT INTO '{self.catalog}.fraud.transactions' SELECT * FROM transactions"
        )
        temp.close()

    def read_applications(self) -> pl.DataFrame:
        self._ensure_connection()
        return self._conn.execute(
            f"SELECT * FROM {self.catalog}.fraud.applications ORDER BY application_id"
        ).pl()

    def read_transactions(self) -> pl.DataFrame:
        self._ensure_connection()
        return self._conn.execute(
            f"SELECT * FROM {self.catalog}.fraud.transactions ORDER BY application_id, transaction_time"
        ).pl()

    def get_last_processed_id(self) -> int:
        self._ensure_connection()
        result = self._conn.execute(
            f"SELECT value FROM {self.catalog}.fraud.pipeline_state WHERE key = 'last_processed_id'"
        ).fetchone()
        if result:
            return int(result[0])
        return 0

    def set_last_processed_id(self, app_id: int) -> None:
        self._ensure_connection()
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self.catalog}.fraud.pipeline_state (key, value) "
            f"VALUES ('last_processed_id', '{app_id}')"
        )


def get_olap_store(name: str = "duckdb") -> OLAPStore:
    name = name or os.environ.get("OLAP_STORE", "duckdb").lower()

    if name == "databricks":
        return DatabricksStore()
    return DuckDBStore()