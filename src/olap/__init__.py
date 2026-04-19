import os
from abc import ABC, abstractmethod
from datetime import datetime
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
    def __init__(self, warehouse_id: str | None = None):
        self.warehouse_id = warehouse_id or os.environ.get("DATABRICKS_WAREHOUSE_ID")
        self._spark = None

    def _ensure_spark(self):
        if self._spark is None:
            try:
                from pyspark.sql import SparkSession

                self._spark = (
                    SparkSession.builder.appName("fraud-etl")
                    .config("spark.sql.warehouse.dir", "/mnt/warehouse")
                    .getOrCreate()
                )
            except ImportError:
                raise ImportError("pyspark not installed. Install with: pip install pyspark")

    def write_applications(self, df: pl.DataFrame) -> None:
        self._ensure_spark()
        self._spark.createDataFrame(df.to_pandas()).write.mode("overwrite").saveAsTable("applications")

    def write_transactions(self, df: pl.DataFrame) -> None:
        self._ensure_spark()
        self._spark.createDataFrame(df.to_pandas()).write.mode("overwrite").saveAsTable("transactions")

    def read_applications(self) -> pl.DataFrame:
        self._ensure_spark()
        pdf = self._spark.read.table("applications").toPandas()
        return pl.from_pandas(pdf)

    def read_transactions(self) -> pl.DataFrame:
        self._ensure_spark()
        pdf = self._spark.read.table("transactions").toPandas()
        return pl.from_pandas(pdf)

    def get_last_processed_id(self) -> int:
        self._ensure_spark()
        df = self._spark.read.table("pipeline_state").filter("key = 'last_processed_id'").toPandas()
        if df.empty:
            return 0
        return int(df.iloc[0]["value"])

    def set_last_processed_id(self, app_id: int) -> None:
        self._ensure_spark()
        self._spark.createDataFrame([{"key": "last_processed_id", "value": str(app_id)}]).write.mode("append").saveAsTable("pipeline_state")


def get_olap_store(name: str | None = None) -> OLAPStore:
    name = name or os.environ.get("OLAP_STORE", "duckdb").lower()

    if name == "databricks":
        return DatabricksStore()
    return DuckDBStore()