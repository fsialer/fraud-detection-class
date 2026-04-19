import os
import logging
from datetime import datetime
from typing import Optional

import polars as pl
import psycopg2

from src.fraud import run_all, REJECTION_THRESHOLD
from src.olap import get_olap_store, OLAPStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_neon_connection():
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def extract_from_neon(conn) -> tuple[pl.DataFrame, pl.DataFrame]:
    applications = pl.read_database("SELECT * FROM credit_applications", connection=conn)
    transactions = pl.read_database("SELECT * FROM transactions", connection=conn)
    return applications, transactions


def load_to_olap(olap: OLAPStore, applications: pl.DataFrame, transactions: pl.DataFrame):
    olap.write_applications(applications)
    olap.write_transactions(transactions)
    logger.info(f"Loaded {len(applications)} applications and {len(transactions)} transactions to OLAP")


def run_fraud_pipeline(olap: OLAPStore) -> list[dict]:
    applications = olap.read_applications()
    transactions = olap.read_transactions()

    logger.info(f"Running fraud rules on {len(applications)} applications")
    fraud_flags = run_all(transactions, applications)
    logger.info(f"Fraud rules generated {len(fraud_flags)} flags")
    return fraud_flags


def write_results_to_neon(conn, fraud_flags: list[dict]):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fraud_results")

    for flag in fraud_flags:
        cursor.execute(
            """
            INSERT INTO fraud_results (application_id, rule_name, triggered, score, details)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                flag["application_id"],
                flag["rule_name"],
                flag["triggered"],
                flag["score"],
                flag["details"],
            ),
        )
    conn.commit()
    logger.info(f"Wrote {len(fraud_flags)} fraud results to Neon")


def update_application_status(conn, olap: OLAPStore):
    cursor = conn.cursor()

    df = olap.read_applications()
    if df.is_empty():
        return

    app_ids = df["application_id"].to_list()

    for app_id in app_ids:
        cursor.execute(
            """
            SELECT COUNT(*) FROM fraud_results
            WHERE application_id = %s AND triggered = true
            """,
            (app_id,),
        )
        triggered_count = cursor.fetchone()[0]

        decision = "rejected" if triggered_count >= REJECTION_THRESHOLD else "approved"
        cursor.execute(
            "UPDATE credit_applications SET status = %s WHERE id = %s",
            (decision, app_id),
        )

    conn.commit()
    logger.info(f"Updated statuses for {len(app_ids)} applications")


def run_etl(incremental: bool = False, olap_store: Optional[OLAPStore] = None):
    logger.info("Starting ETL pipeline")

    if olap_store is None:
        olap_store = get_olap_store()

    conn = get_neon_connection()

    try:
        applications, transactions = extract_from_neon(conn)
        logger.info(f"Extracted {len(applications)} applications and {len(transactions)} transactions from Neon")

        if incremental:
            last_id = olap_store.get_last_processed_id()
            applications = applications.filter(pl.col("application_id") > last_id)
            logger.info(f"Incremental mode: processing {len(applications)} new applications")

            if applications.is_empty():
                logger.info("No new applications to process")
                return

        load_to_olap(olap_store, applications, transactions)

        fraud_flags = run_fraud_pipeline(olap_store)

        write_results_to_neon(conn, fraud_flags)

        update_application_status(conn, olap_store)

        if not applications.is_empty():
            max_id = applications["application_id"].max()
            olap_store.set_last_processed_id(max_id)

        logger.info("ETL pipeline completed successfully")

    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    incremental = "--incremental" in sys.argv
    run_etl(incremental=incremental)