"""
Bank Transaction Simulator
===========================
Generates realistic bank/payment transactions with fraud patterns.
Simulates 1000 unique users with behavioral profiles.

Fraud patterns injected:
  1. Velocity abuse       - too many transactions too fast
  2. Geographic anomaly   - impossible location jumps
  3. Amount spike         - way above user's normal spending
  4. Merchant mismatch    - unusual category for this user
  5. Late night fraud     - 2am-5am transactions
  6. Card not present     - online fraud pattern
  7. Failed attempts      - testing stolen card details

Run: python simulator/generate_transactions.py
"""
import csv
import random
import uuid
import os
import math
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FILE        = "data/raw/transactions.csv"
NUM_USERS          = 1000      # simulate 1000 unique bank customers
NUM_TRANSACTIONS   = 10000     # total transactions over 90 days
FRAUD_RATE         = 0.02      # 2% fraud — realistic for banking industry

# Merchant categories with typical legitimate vs fraud weights
LEGITIMATE_MERCHANTS = [
    "grocery", "restaurant", "gas_station", "pharmacy",
    "clothing", "utilities", "healthcare", "entertainment"
]
FRAUD_MERCHANTS = [
    "electronics", "jewelry", "gift_cards",
    "crypto_exchange", "wire_transfer", "luxury_goods"
]
ALL_MERCHANTS = LEGITIMATE_MERCHANTS + FRAUD_MERCHANTS


# ─────────────────────────────────────────────────────────────────────────────
# USER PROFILE GENERATOR
# Each user has a behavioral fingerprint — legitimate transactions
# stay close to it, fraud transactions violate it
# ─────────────────────────────────────────────────────────────────────────────
def generate_user_profiles(num_users):
    """
    Creates a behavioral profile for each user.
    In real banks this comes from 6-12 months of transaction history.
    We define it upfront and simulate transactions around it.
    """
    profiles = {}
    for user_id in range(1, num_users + 1):
        profiles[user_id] = {
            "user_id"            : user_id,
            # Average spend — ranges from budget ($20) to wealthy ($500)
            "avg_transaction_amt": round(random.uniform(20, 500), 2),
            # Home location — US cities (lat, lon)
            "home_lat"           : round(random.uniform(25.0, 48.0), 4),
            "home_lon"           : round(random.uniform(-122.0, -71.0), 4),
            # Preferred merchants — each user has 3-4 they use regularly
            "preferred_merchants": random.sample(LEGITIMATE_MERCHANTS, 3),
            # Active hours — when they normally transact
            "active_hours"       : random.choice([
                range(7, 22),    # typical 7am-10pm person
                range(9, 18),    # 9-5 office worker
                range(6, 14),    # early bird
            ])
        }
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE CALCULATOR
# Haversine formula — calculates real geographic distance between two points
# Used to detect "impossible" location jumps in fraud
# ─────────────────────────────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# VELOCITY TRACKER
# Tracks how many transactions each user made recently
# This is what banks call "velocity checks" — critical fraud signal
# ─────────────────────────────────────────────────────────────────────────────
class VelocityTracker:
    def __init__(self):
        # user_id → list of transaction timestamps
        self.user_transactions = {}

    def add(self, user_id, timestamp):
        if user_id not in self.user_transactions:
            self.user_transactions[user_id] = []
        self.user_transactions[user_id].append(timestamp)

    def count_last_n_hours(self, user_id, timestamp, hours):
        if user_id not in self.user_transactions:
            return 0
        cutoff = timestamp - timedelta(hours=hours)
        return sum(
            1 for t in self.user_transactions[user_id]
            if cutoff <= t <= timestamp
        )


# ─────────────────────────────────────────────────────────────────────────────
# LEGITIMATE TRANSACTION GENERATOR
# Stays close to the user's behavioral profile
# ─────────────────────────────────────────────────────────────────────────────
def legitimate_transaction(user_id, profile, timestamp, velocity):
    # Amount: within 2x of user's average (normal variation)
    amount = round(
        abs(random.gauss(profile["avg_transaction_amt"],
                         profile["avg_transaction_amt"] * 0.5)), 2
    )

    # Location: close to home (within 50km normally)
    lat_offset = random.uniform(-0.3, 0.3)
    lon_offset = random.uniform(-0.3, 0.3)
    txn_lat = profile["home_lat"] + lat_offset
    txn_lon = profile["home_lon"] + lon_offset
    distance = haversine_distance(
        profile["home_lat"], profile["home_lon"], txn_lat, txn_lon
    )

    # Merchant: from their preferred list mostly
    merchant = random.choices(
        profile["preferred_merchants"] + LEGITIMATE_MERCHANTS,
        weights=[3] * len(profile["preferred_merchants"]) + [1] * len(LEGITIMATE_MERCHANTS),
        k=1
    )[0]

    velocity.add(user_id, timestamp)

    return {
        "transaction_id"          : str(uuid.uuid4())[:16],
        "user_id"                 : user_id,
        "timestamp"               : timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "amount"                  : amount,
        "merchant_category"       : merchant,
        "is_international"        : 0,
        "distance_from_home_km"   : distance,
        "card_present"            : random.choices([1, 0], weights=[70, 30])[0],
        "transactions_last_1h"    : velocity.count_last_n_hours(user_id, timestamp, 1),
        "transactions_last_24h"   : velocity.count_last_n_hours(user_id, timestamp, 24),
        "amount_vs_user_avg"      : round(amount / profile["avg_transaction_amt"], 3),
        "hour_of_day"             : timestamp.hour,
        "failed_attempts"         : random.choices([0, 1], weights=[90, 10])[0],
        "device_match"            : random.choices([1, 0], weights=[85, 15])[0],
        "is_fraud"                : 0
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD TRANSACTION GENERATOR
# Deliberately violates user's behavioral profile
# Each fraud type mimics a real-world attack pattern
# ─────────────────────────────────────────────────────────────────────────────
def fraud_transaction(user_id, profile, timestamp, velocity):
    fraud_type = random.choice([
        "velocity_abuse",
        "geographic_anomaly",
        "amount_spike",
        "merchant_mismatch",
        "late_night",
        "card_not_present"
    ])

    # Base defaults
    base = {
        "transaction_id"    : str(uuid.uuid4())[:16],
        "user_id"           : user_id,
        "timestamp"         : timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "hour_of_day"       : timestamp.hour,
        "merchant_category" : random.choice(FRAUD_MERCHANTS),
        "is_international"  : 0,
        "is_fraud"          : 1
    }

    if fraud_type == "velocity_abuse":
        # Fraudster makes many small transactions rapidly
        # Testing card: $1, $5, $10 then big purchase
        amount = round(random.uniform(1, 50), 2)
        base.update({
            "amount"               : amount,
            "distance_from_home_km": round(random.uniform(0, 20), 2),
            "card_present"         : 0,
            "transactions_last_1h" : random.randint(8, 20),  # SPIKE
            "transactions_last_24h": random.randint(20, 50),
            "amount_vs_user_avg"   : round(amount / profile["avg_transaction_amt"], 3),
            "failed_attempts"      : random.randint(3, 8),   # tested card multiple times
            "device_match"         : 0,
        })

    elif fraud_type == "geographic_anomaly":
        # Transaction happening thousands of km from home
        # While real card is in user's wallet
        amount = round(random.uniform(
            profile["avg_transaction_amt"] * 2,
            profile["avg_transaction_amt"] * 8
        ), 2)
        # Random location far from home — international
        fraud_lat = round(random.uniform(-50, 70), 4)
        fraud_lon = round(random.uniform(-150, 150), 4)
        distance  = haversine_distance(
            profile["home_lat"], profile["home_lon"], fraud_lat, fraud_lon
        )
        base.update({
            "amount"               : amount,
            "is_international"     : 1,
            "distance_from_home_km": distance,   # HUGE distance
            "card_present"         : random.choice([0, 1]),
            "transactions_last_1h" : random.randint(1, 3),
            "transactions_last_24h": random.randint(2, 8),
            "amount_vs_user_avg"   : round(amount / profile["avg_transaction_amt"], 3),
            "failed_attempts"      : random.randint(0, 2),
            "device_match"         : 0,           # unknown device/location
        })

    elif fraud_type == "amount_spike":
        # Single massive transaction — way above user's normal spending
        amount = round(random.uniform(
            profile["avg_transaction_amt"] * 10,  # 10x to 50x normal!
            profile["avg_transaction_amt"] * 50
        ), 2)
        base.update({
            "amount"               : amount,
            "distance_from_home_km": round(random.uniform(5, 100), 2),
            "card_present"         : 0,
            "transactions_last_1h" : random.randint(1, 3),
            "transactions_last_24h": random.randint(1, 5),
            "amount_vs_user_avg"   : round(amount / profile["avg_transaction_amt"], 3),
            "failed_attempts"      : random.randint(1, 4),
            "device_match"         : random.choice([0, 1]),
        })

    elif fraud_type == "merchant_mismatch":
        # User who only buys groceries suddenly buys $3000 of jewelry
        amount = round(random.uniform(
            profile["avg_transaction_amt"] * 3,
            profile["avg_transaction_amt"] * 15
        ), 2)
        base.update({
            "merchant_category"    : random.choice(["jewelry", "electronics", "luxury_goods"]),
            "amount"               : amount,
            "distance_from_home_km": round(random.uniform(10, 200), 2),
            "card_present"         : random.choice([0, 1]),
            "transactions_last_1h" : random.randint(1, 4),
            "transactions_last_24h": random.randint(1, 8),
            "amount_vs_user_avg"   : round(amount / profile["avg_transaction_amt"], 3),
            "failed_attempts"      : random.randint(0, 3),
            "device_match"         : random.choice([0, 1]),
        })

    elif fraud_type == "late_night":
        # 2am-5am transaction for user who never shops at night
        fraud_hour = random.randint(2, 5)
        fraud_ts   = timestamp.replace(hour=fraud_hour)
        amount     = round(random.uniform(
            profile["avg_transaction_amt"] * 2,
            profile["avg_transaction_amt"] * 10
        ), 2)
        base.update({
            "timestamp"            : fraud_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "hour_of_day"          : fraud_hour,
            "amount"               : amount,
            "distance_from_home_km": round(random.uniform(0, 50), 2),
            "card_present"         : 0,
            "transactions_last_1h" : random.randint(1, 5),
            "transactions_last_24h": random.randint(1, 8),
            "amount_vs_user_avg"   : round(amount / profile["avg_transaction_amt"], 3),
            "failed_attempts"      : random.randint(1, 5),
            "device_match"         : 0,
        })

    elif fraud_type == "card_not_present":
        # Online fraud — fraudster has card number but not physical card
        amount = round(random.uniform(
            profile["avg_transaction_amt"] * 1.5,
            profile["avg_transaction_amt"] * 12
        ), 2)
        base.update({
            "amount"               : amount,
            "distance_from_home_km": round(random.uniform(100, 5000), 2),
            "card_present"         : 0,           # ALWAYS online
            "is_international"     : random.choice([0, 1]),
            "transactions_last_1h" : random.randint(2, 8),
            "transactions_last_24h": random.randint(3, 15),
            "amount_vs_user_avg"   : round(amount / profile["avg_transaction_amt"], 3),
            "failed_attempts"      : random.randint(2, 6),
            "device_match"         : 0,           # unknown device
        })

    velocity.add(user_id, timestamp)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_transactions():
    os.makedirs("data/raw", exist_ok=True)

    print("Generating user profiles...")
    profiles = generate_user_profiles(NUM_USERS)
    print(f"  Created {NUM_USERS} unique user behavioral profiles")

    velocity  = VelocityTracker()
    records   = []
    start_time = datetime.now() - timedelta(days=90)  # 90 days history

    print(f"\nGenerating {NUM_TRANSACTIONS} transactions...")

    for i in range(NUM_TRANSACTIONS):
        # Progress every 1000 records
        if i % 1000 == 0:
            print(f"  {i}/{NUM_TRANSACTIONS} transactions generated...")

        # Spread transactions over 90 days with realistic time jitter
        timestamp = start_time + timedelta(
            seconds = i * (90 * 24 * 3600 / NUM_TRANSACTIONS)
                      + random.randint(-300, 300)
        )

        user_id = random.randint(1, NUM_USERS)
        profile = profiles[user_id]

        if random.random() < FRAUD_RATE:
            records.append(fraud_transaction(user_id, profile, timestamp, velocity))
        else:
            records.append(legitimate_transaction(user_id, profile, timestamp, velocity))

    # Write CSV
    fieldnames = [
        "transaction_id", "user_id", "timestamp", "amount",
        "merchant_category", "is_international", "distance_from_home_km",
        "card_present", "transactions_last_1h", "transactions_last_24h",
        "amount_vs_user_avg", "hour_of_day", "failed_attempts",
        "device_match", "is_fraud"
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Summary
    fraud_count = sum(1 for r in records if r["is_fraud"] == 1)
    legit_count = NUM_TRANSACTIONS - fraud_count

    print(f"\nDone! File saved: {OUTPUT_FILE}")
    print(f"  Total transactions : {NUM_TRANSACTIONS}")
    print(f"  Legitimate         : {legit_count} ({legit_count/NUM_TRANSACTIONS*100:.1f}%)")
    print(f"  Fraud              : {fraud_count} ({fraud_count/NUM_TRANSACTIONS*100:.1f}%)")

    print(f"\nFraud type breakdown:")
    fraud_types = ["velocity_abuse", "geographic_anomaly", "amount_spike",
                   "merchant_mismatch", "late_night", "card_not_present"]
    print(f"  6 fraud attack patterns injected across {fraud_count} transactions")

    print(f"\nSample legitimate transaction:")
    legit = next(r for r in records if r["is_fraud"] == 0)
    for k, v in legit.items():
        print(f"  {k:30s}: {v}")

    print(f"\nSample FRAUD transaction:")
    fraud = next(r for r in records if r["is_fraud"] == 1)
    for k, v in fraud.items():
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    generate_transactions()