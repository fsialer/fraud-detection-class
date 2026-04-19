import polars as pl
from datetime import datetime
import pytest
from src.fraud.rules import (
    velocity_check,
    income_ratio,
    geo_anomaly,
    high_risk_merchant,
    unusual_hours,
)


class TestVelocityCheck:
    def test_triggers_with_four_transactions_in_one_hour(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1, 1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 10, 20),
                datetime(2024, 1, 1, 10, 40),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 11, 20),
            ],
            'amount': [100.0] * 5,
            'merchant': ['m'] * 5,
            'category': ['retail'] * 5,
            'location_country': ['US'] * 5,
        })
        results = velocity_check(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 1
        assert triggered[0]['application_id'] == 1
        assert triggered[0]['score'] == 5.0

    def test_no_trigger_with_transactions_spaced_out(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 2, 10, 0),
            ],
            'amount': [100.0] * 2,
            'merchant': ['m'] * 2,
            'category': ['retail'] * 2,
            'location_country': ['US'] * 2,
        })
        results = velocity_check(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_exactly_three_transactions_no_trigger(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 12, 0),
            ],
            'amount': [100.0] * 3,
            'merchant': ['m'] * 3,
            'category': ['retail'] * 3,
            'location_country': ['US'] * 3,
        })
        results = velocity_check(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_empty_transactions(self):
        transactions = pl.DataFrame({
            'application_id': [],
            'transaction_time': [],
            'amount': [],
            'merchant': [],
            'category': [],
            'location_country': [],
        })
        results = velocity_check(transactions)
        assert results == []


class TestIncomeRatio:
    def test_trigger_when_spending_exceeds_half_of_income(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1],
            'transaction_time': [datetime(2024, 1, 1)] * 3,
            'amount': [3000.0, 3000.0, 3000.0],
            'merchant': ['m'] * 3,
            'category': ['retail'] * 3,
            'location_country': ['US'] * 3,
        })
        applications = pl.DataFrame({
            'application_id': [1],
            'annual_income': [10000.0],
        })
        results = income_ratio(transactions, applications)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 1
        assert triggered[0]['score'] == 90.0

    def test_no_trigger_when_spending_below_half_of_income(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1],
            'transaction_time': [datetime(2024, 1, 1)] * 2,
            'amount': [1000.0, 1000.0],
            'merchant': ['m'] * 2,
            'category': ['retail'] * 2,
            'location_country': ['US'] * 2,
        })
        applications = pl.DataFrame({
            'application_id': [1],
            'annual_income': [10000.0],
        })
        results = income_ratio(transactions, applications)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_exactly_fifty_percent_no_trigger(self):
        transactions = pl.DataFrame({
            'application_id': [1],
            'transaction_time': [datetime(2024, 1, 1)],
            'amount': [5000.0],
            'merchant': ['m'],
            'category': ['retail'],
            'location_country': ['US'],
        })
        applications = pl.DataFrame({
            'application_id': [1],
            'annual_income': [10000.0],
        })
        results = income_ratio(transactions, applications)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0


class TestGeoAnomaly:
    def test_trigger_with_three_countries_in_forty_eight_hours(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 13, 0),
            ],
            'amount': [100.0] * 4,
            'merchant': ['m'] * 4,
            'category': ['retail'] * 4,
            'location_country': ['US', 'MX', 'CA', 'UK'],
        })
        results = geo_anomaly(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 1
        assert triggered[0]['score'] == 4.0

    def test_no_trigger_with_single_country(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1],
            'transaction_time': [datetime(2024, 1, 1)] * 3,
            'amount': [100.0] * 3,
            'merchant': ['m'] * 3,
            'category': ['retail'] * 3,
            'location_country': ['US', 'US', 'US'],
        })
        results = geo_anomaly(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_two_countries_no_trigger(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
            ],
            'amount': [100.0] * 2,
            'merchant': ['m'] * 2,
            'category': ['retail'] * 2,
            'location_country': ['US', 'MX'],
        })
        results = geo_anomaly(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_empty_transactions(self):
        transactions = pl.DataFrame({
            'application_id': [],
            'transaction_time': [],
            'amount': [],
            'merchant': [],
            'category': [],
            'location_country': [],
        })
        results = geo_anomaly(transactions)
        assert results == []


class TestHighRiskMerchant:
    def test_trigger_when_high_risk_exceeds_thirty_percent(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1, 1],
            'transaction_time': [datetime(2024, 1, 1)] * 4,
            'amount': [100.0] * 4,
            'merchant': ['m'] * 4,
            'category': ['gambling', 'crypto', 'retail', 'retail'],
            'location_country': ['US'] * 4,
        })
        results = high_risk_merchant(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 1
        assert triggered[0]['score'] == 50.0

    def test_no_trigger_when_high_risk_below_thirty_percent(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1, 1, 1, 1, 1],
            'transaction_time': [datetime(2024, 1, 1)] * 7,
            'amount': [100.0] * 7,
            'merchant': ['m'] * 7,
            'category': ['gambling', 'retail', 'retail', 'retail', 'retail', 'retail', 'retail'],
            'location_country': ['US'] * 7,
        })
        results = high_risk_merchant(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_exactly_thirty_percent_no_trigger(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'transaction_time': [datetime(2024, 1, 1)] * 10,
            'amount': [100.0] * 10,
            'merchant': ['m'] * 10,
            'category': ['gambling', 'crypto', 'cash'] + ['retail'] * 7,
            'location_country': ['US'] * 10,
        })
        results = high_risk_merchant(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0


class TestUnusualHours:
    def test_trigger_when_majority_between_midnight_and_5am(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 2, 0),
                datetime(2024, 1, 1, 3, 0),
                datetime(2024, 1, 1, 12, 0),
            ],
            'amount': [100.0] * 3,
            'merchant': ['m'] * 3,
            'category': ['retail'] * 3,
            'location_country': ['US'] * 3,
        })
        results = unusual_hours(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 1
        assert triggered[0]['score'] == pytest.approx(66.67, rel=0.01)

    def test_no_trigger_when_majority_during_normal_hours(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 1, 18, 0),
            ],
            'amount': [100.0] * 3,
            'merchant': ['m'] * 3,
            'category': ['retail'] * 3,
            'location_country': ['US'] * 3,
        })
        results = unusual_hours(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_exactly_fifty_percent_no_trigger(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1],
            'transaction_time': [
                datetime(2024, 1, 1, 3, 0),
                datetime(2024, 1, 1, 12, 0),
            ],
            'amount': [100.0] * 2,
            'merchant': ['m'] * 2,
            'category': ['retail'] * 2,
            'location_country': ['US'] * 2,
        })
        results = unusual_hours(transactions)
        triggered = [r for r in results if r['triggered']]
        assert len(triggered) == 0

    def test_edge_empty_transactions(self):
        transactions = pl.DataFrame({
            'application_id': [],
            'transaction_time': [],
            'amount': [],
            'merchant': [],
            'category': [],
            'location_country': [],
        })
        results = unusual_hours(transactions)
        assert results == []


class TestMultipleApplicants:
    def test_velocity_check_multiple_applicants(self):
        transactions = pl.DataFrame({
            'application_id': [1, 1, 1, 1, 1, 2, 2],
            'transaction_time': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 13, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 3, 10, 0),
            ],
            'amount': [100.0] * 7,
            'merchant': ['m'] * 7,
            'category': ['retail'] * 7,
            'location_country': ['US'] * 7,
        })
        results = velocity_check(transactions)
        app1_triggered = [r for r in results if r['application_id'] == 1 and r['triggered']]
        app2_triggered = [r for r in results if r['application_id'] == 2 and r['triggered']]
        assert len(app1_triggered) == 1
        assert len(app2_triggered) == 0