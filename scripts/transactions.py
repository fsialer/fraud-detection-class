"""
Simulates a bank feed: inserts card transactions every ~1 second
and new credit applications every ~5 seconds.
Run this in the background to generate live data.

Usage:
    uv run python scripts/transactions.py
"""

import os
import random
import time

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

MERCHANTS = {
    "groceries": ["Walmart", "Kroger", "Whole Foods", "Costco", "Trader Joes"],
    "electronics": ["BestBuy", "Apple Store", "Newegg", "MicroCenter", "Amazon"],
    "food": ["McDonalds", "Chipotle", "Starbucks", "Subway", "Panera"],
    "fuel": ["Shell Gas", "Chevron", "BP", "ExxonMobil"],
    "entertainment": ["Netflix", "Spotify", "AMC Theaters", "Steam"],
    "clothing": ["Nordstrom", "Target", "Zara", "H&M", "Gap"],
    "gambling": ["Online Casino", "DraftKings", "FanDuel"],
    "crypto": ["CryptoExchange", "Coinbase", "Binance"],
    "transfer": ["Wire Transfer", "Zelle", "Venmo"],
    "cash": ["Cash Advance", "ATM Withdrawal"],
    "luxury": ["Harrods", "Gucci", "Louis Vuitton"],
    "transport": ["Uber", "Lyft", "Delta Airlines"],
}

COUNTRIES = ["US", "US", "US", "US", "US", "GB", "FR", "DE", "MX", "NG", "RU", "JP"]

FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "Logan", "Mia", "Lucas", "Charlotte", "Alexander", "Amelia",
    "Daniel", "Harper", "Matthew", "Evelyn", "Aiden", "Luna", "Henry",
    "Camila", "Sebastian", "Gianna", "Jack", "Aria", "Owen", "Ella", "Samuel",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson",
]

EMPLOYMENT_STATUSES = ["employed", "employed", "employed", "self-employed", "unemployed"]

CATEGORIES_WEIGHTED = (
    ["groceries"] * 20
    + ["food"] * 15
    + ["fuel"] * 10
    + ["electronics"] * 10
    + ["entertainment"] * 10
    + ["clothing"] * 8
    + ["transport"] * 7
    + ["transfer"] * 5
    + ["gambling"] * 5
    + ["crypto"] * 4
    + ["cash"] * 3
    + ["luxury"] * 3
)


def get_applicant_ids(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM credit_applications")
        return [row[0] for row in cur.fetchall()]


def generate_transaction(applicant_id):
    category = random.choice(CATEGORIES_WEIGHTED)
    merchant = random.choice(MERCHANTS[category])
    country = random.choice(COUNTRIES)
    is_online = random.random() < 0.4

    # Amount varies by category
    amount_ranges = {
        "groceries": (15, 200),
        "food": (8, 60),
        "fuel": (25, 80),
        "electronics": (50, 3000),
        "entertainment": (10, 100),
        "clothing": (20, 500),
        "gambling": (50, 5000),
        "crypto": (100, 5000),
        "transfer": (100, 10000),
        "cash": (20, 1000),
        "luxury": (200, 5000),
        "transport": (15, 800),
    }
    lo, hi = amount_ranges[category]
    amount = round(random.uniform(lo, hi), 2)

    return {
        "applicant_id": applicant_id,
        "amount": amount,
        "merchant": merchant,
        "category": category,
        "location_country": country,
        "is_online": is_online,
    }


def generate_application():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@email.com"
    ssn_last4 = f"{random.randint(1000, 9999)}"
    annual_income = round(random.uniform(30000, 200000), 2)
    requested_amount = round(random.uniform(5000, 75000), 2)
    employment_status = random.choice(EMPLOYMENT_STATUSES)

    return {
        "applicant_name": name,
        "email": email,
        "ssn_last4": ssn_last4,
        "annual_income": annual_income,
        "requested_amount": requested_amount,
        "employment_status": employment_status,
    }


def insert_application(conn, app):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO credit_applications (applicant_name, email, ssn_last4, annual_income, requested_amount, employment_status)
            VALUES (%(applicant_name)s, %(email)s, %(ssn_last4)s, %(annual_income)s, %(requested_amount)s, %(employment_status)s)
            RETURNING id
            """,
            app,
        )
        conn.commit()
        return cur.fetchone()[0]


def insert_transaction(conn, tx):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO transactions (applicant_id, amount, merchant, category, location_country, is_online)
            VALUES (%(applicant_id)s, %(amount)s, %(merchant)s, %(category)s, %(location_country)s, %(is_online)s)
            RETURNING id
            """,
            tx,
        )
        conn.commit()
        return cur.fetchone()[0]


def main():
    conn = psycopg2.connect(DATABASE_URL)
    applicant_ids = get_applicant_ids(conn)

    if not applicant_ids:
        print("No applicants found — will create some.")

    print("Simulating bank feed: transactions every ~1s, new applications every ~5s")
    print("Press Ctrl+C to stop.\n")

    tx_count = 0
    app_count = 0
    last_application_time = time.time()

    try:
        while True:
            # New credit application every ~5 seconds
            if time.time() - last_application_time >= 5:
                app = generate_application()
                app_id = insert_application(conn, app)
                applicant_ids.append(app_id)
                app_count += 1
                print(
                    f"  [NEW APP #{app_id}] {app['applicant_name']:<20} "
                    f"${app['requested_amount']:>10,.2f} ({app['employment_status']})"
                )
                last_application_time = time.time()

            # Transaction every ~1 second
            if applicant_ids:
                applicant_id = random.choice(applicant_ids)
                tx = generate_transaction(applicant_id)
                insert_transaction(conn, tx)
                tx_count += 1
                print(
                    f"  [TX {tx_count}] applicant={applicant_id:<4} "
                    f"${tx['amount']:>9,.2f} {tx['category']:<14} "
                    f"{tx['merchant']:<20} {tx['location_country']}"
                )

            time.sleep(random.uniform(0.5, 1.5))
    except KeyboardInterrupt:
        print(f"\nStopped. Inserted {tx_count} transactions, {app_count} applications.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
