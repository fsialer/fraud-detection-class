import polars as pl
from src.fraud.rules import (
    velocity_check,
    income_ratio,
    geo_anomaly,
    high_risk_merchant,
    unusual_hours,
)

REJECTION_THRESHOLD = 2


def run_all(transactions: pl.DataFrame, applications: pl.DataFrame) -> list[dict]:
    """
    Run all fraud detection rules and return combined results.
    """
    results = []
    results.extend(velocity_check(transactions))
    results.extend(income_ratio(transactions, applications))
    results.extend(geo_anomaly(transactions))
    results.extend(high_risk_merchant(transactions))
    results.extend(unusual_hours(transactions))
    return results


def to_dataframe(flags: list[dict]) -> pl.DataFrame:
    """
    Convert fraud flags list to a Polars DataFrame.
    """
    return pl.DataFrame(flags)


def decide_applications(flags: list[dict]) -> pl.DataFrame:
    """
    Decide application status based on fraud detection results.
    """
    df = pl.DataFrame(flags)

    summary = df.group_by('application_id').agg(
        pl.col('triggered').sum().alias('rules_triggered'),
    )

    summary = summary.with_columns(
        pl.when(pl.col('rules_triggered') >= REJECTION_THRESHOLD)
        .then(pl.lit('rejected'))
        .otherwise(pl.lit('approved'))
        .alias('decision')
    )

    return summary