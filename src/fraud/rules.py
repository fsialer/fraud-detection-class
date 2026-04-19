import polars as pl
from datetime import timedelta


def velocity_check(transactions: pl.DataFrame) -> list[dict]:
    """
    Flag applicants with >3 transactions in any 4-hour window.
    """
    results = []
    if transactions.is_empty():
        return results

    app_ids = transactions.unique('application_id')['application_id'].to_list()

    for app_id in app_ids:
        app_txns = transactions.filter(pl.col('application_id') == app_id).sort('transaction_time')

        if app_txns.height <= 3:
            results.append({
                'application_id': app_id,
                'rule_name': 'velocity_check',
                'triggered': False,
                'score': 0.0,
                'details': f'Transactions: {app_txns.height}'
            })
            continue

        txn_times = app_txns['transaction_time'].to_list()
        max_count = 0

        for i in range(len(txn_times)):
            window_start = txn_times[i]
            window_end = window_start + timedelta(hours=4)
            count = sum(1 for t in txn_times if window_start <= t < window_end)
            max_count = max(max_count, count)

        triggered = max_count > 3
        results.append({
            'application_id': app_id,
            'rule_name': 'velocity_check',
            'triggered': triggered,
            'score': float(max_count) if triggered else 0.0,
            'details': f'Max transactions in 4h window: {max_count}'
        })

    return results


def income_ratio(transactions: pl.DataFrame, applications: pl.DataFrame) -> list[dict]:
    """
    Flag applicants whose total spending exceeds 50% of annual income.
    """
    results = []
    if transactions.is_empty() or applications.is_empty():
        return results

    app_ids = applications.unique('application_id')['application_id'].to_list()

    for app_id in app_ids:
        app_txns = transactions.filter(pl.col('application_id') == app_id)
        total_spending = app_txns['amount'].sum()

        income_row = applications.filter(pl.col('application_id') == app_id)
        if income_row.is_empty():
            continue

        annual_income = income_row['annual_income'].to_list()[0]
        ratio = total_spending / annual_income if annual_income > 0 else 0.0

        triggered = ratio > 0.5
        results.append({
            'application_id': app_id,
            'rule_name': 'income_ratio',
            'triggered': triggered,
            'score': round(ratio * 100, 2) if triggered else 0.0,
            'details': f'Spending: ${total_spending:.2f}, Income: ${annual_income:.2f}, Ratio: {ratio:.1%}'
        })

    return results


def geo_anomaly(transactions: pl.DataFrame) -> list[dict]:
    """
    Flag applicants with transactions in 3+ countries within 48 hours.
    """
    results = []
    if transactions.is_empty():
        return results

    app_ids = transactions.unique('application_id')['application_id'].to_list()

    for app_id in app_ids:
        app_txns = transactions.filter(pl.col('application_id') == app_id).sort('transaction_time')

        if app_txns.height < 3:
            results.append({
                'application_id': app_id,
                'rule_name': 'geo_anomaly',
                'triggered': False,
                'score': 0.0,
                'details': f'Transactions: {app_txns.height}'
            })
            continue

        txn_times = app_txns['transaction_time'].to_list()
        countries = app_txns['location_country'].to_list()
        max_countries = 0

        for i in range(len(txn_times)):
            window_start = txn_times[i]
            window_end = window_start + timedelta(hours=48)
            countries_in_window = set(
                countries[j] for j in range(len(txn_times))
                if window_start <= txn_times[j] < window_end
            )
            max_countries = max(max_countries, len(countries_in_window))

        triggered = max_countries >= 3
        results.append({
            'application_id': app_id,
            'rule_name': 'geo_anomaly',
            'triggered': triggered,
            'score': float(max_countries) if triggered else 0.0,
            'details': f'Countries in 48h window: {max_countries}'
        })

    return results


def high_risk_merchant(transactions: pl.DataFrame) -> list[dict]:
    """
    Flag applicants with >30% of transactions in high-risk categories.
    """
    results = []
    if transactions.is_empty():
        return results

    high_risk = ['gambling', 'crypto', 'cash']
    app_ids = transactions.unique('application_id')['application_id'].to_list()

    for app_id in app_ids:
        app_txns = transactions.filter(pl.col('application_id') == app_id)
        total_txns = app_txns.height

        if total_txns == 0:
            continue

        high_risk_txns = app_txns.filter(pl.col('category').is_in(high_risk)).height
        ratio = high_risk_txns / total_txns

        triggered = ratio > 0.3
        results.append({
            'application_id': app_id,
            'rule_name': 'high_risk_merchant',
            'triggered': triggered,
            'score': round(ratio * 100, 2) if triggered else 0.0,
            'details': f'High-risk: {high_risk_txns}/{total_txns} ({ratio:.0%})'
        })

    return results


def unusual_hours(transactions: pl.DataFrame) -> list[dict]:
    """
    Flag applicants with >50% of transactions between midnight and 5AM.
    """
    results = []
    if transactions.is_empty():
        return results

    app_ids = transactions.unique('application_id')['application_id'].to_list()

    for app_id in app_ids:
        app_txns = transactions.filter(pl.col('application_id') == app_id)
        total_txns = app_txns.height

        if total_txns == 0:
            continue

        hours = app_txns['transaction_time'].dt.hour().to_list()
        unusual_count = sum(1 for h in hours if 0 <= h < 5)
        ratio = unusual_count / total_txns

        triggered = ratio > 0.5
        results.append({
            'application_id': app_id,
            'rule_name': 'unusual_hours',
            'triggered': triggered,
            'score': round(ratio * 100, 2) if triggered else 0.0,
            'details': f'Unusual hours: {unusual_count}/{total_txns} ({ratio:.0%})'
        })

    return results