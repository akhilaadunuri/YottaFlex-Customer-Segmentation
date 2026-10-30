
import pandas as pd
import numpy as np
import os
import sys

# ==========================================
# YOTTAFLEX CUSTOMER SEGMENTATION PROJECT
# ==========================================

SEED = 20261022
rng = np.random.default_rng(SEED)

FULL = "--full" in sys.argv
N = 50000 if FULL else 1000
O = 200000 if FULL else 4000
E = 1000000 if FULL else 20000

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Starting YottaFlex project...")

# 1. CUSTOMER MASTER
customer_ids = [f"CUS-{i:06d}" for i in range(1, N+1)]

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "name": [f"Customer_{i}" for i in range(N)],
    "city": rng.choice(
        ["Hyderabad", "Mumbai", "Delhi", "Chennai"], N
    ),
    "signup_date": pd.Timestamp("2024-01-01") +
        pd.to_timedelta(rng.integers(0, 1000, N), unit="D"),
    "status": rng.choice(
        ["Active", "New", "Dormant"], N
    )
})

# 2. PRODUCT MASTER
product_ids = [f"PRD-{i:04d}" for i in range(1, 51)]

products = pd.DataFrame({
    "product_id": product_ids,
    "product_name": [f"Product_{i}" for i in range(1, 51)],
    "category": rng.choice(
        ["Electronics", "Fashion", "Home"], 50
    ),
    "price": rng.integers(100, 5000, 50)
})

# 3. ORDERS
order_ids = [f"ORD-2026-{i:06d}" for i in range(1, O+1)]

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": rng.choice(customer_ids, O),
    "order_date": pd.Timestamp("2025-01-01") +
        pd.to_timedelta(rng.integers(0, 668, O), unit="D"),
    "status": rng.choice(
        ["Delivered", "Returned", "Pending"],
        O, p=[0.85, 0.10, 0.05]
    )
})

# 4. ORDER LINES
L = O * 2

order_lines = pd.DataFrame({
    "order_line_id": [f"LINE-{i:07d}" for i in range(1, L+1)],
    "order_id": rng.choice(order_ids, L),
    "product_id": rng.choice(product_ids, L),
    "quantity": rng.integers(1, 4, L)
})

order_lines = order_lines.merge(
    products[["product_id", "price"]], on="product_id"
)

order_lines["line_total"] = (
    order_lines["quantity"] * order_lines["price"]
)

# 5. ENGAGEMENT EVENTS
events = pd.DataFrame({
    "event_id": [f"EVT-{i:08d}" for i in range(1, E+1)],
    "customer_id": rng.choice(customer_ids, E),
    "event_type": rng.choice(
        ["WebsiteVisit", "EmailOpen", "Click", "AppLogin"], E
    ),
    "event_date": pd.Timestamp("2026-01-01") +
        pd.to_timedelta(rng.integers(0, 303, E), unit="D")
})

# 6. SUPPORT CASES
S = max(100, N // 2)

support = pd.DataFrame({
    "case_id": [f"CASE-{i:06d}" for i in range(1, S+1)],
    "customer_id": rng.choice(customer_ids, S),
    "issue": rng.choice(["Refund", "Delivery", "Product"], S),
    "service_cost": rng.integers(50, 2000, S),
    "status": rng.choice(["Open", "Closed"], S)
})

# 7. CONSENT
consent = pd.DataFrame({
    "consent_id": [f"CON-{i:06d}" for i in range(1, N+1)],
    "customer_id": customer_ids,
    "marketing_allowed": rng.choice(
        [True, False], N, p=[0.8, 0.2]
    ),
    "channel": rng.choice(["Email", "SMS"], N),
    "effective_date": "2026-01-01"
})

# 8. RFM ANALYSIS
delivered = orders[orders["status"] == "Delivered"]

order_totals = order_lines.groupby("order_id")[
    "line_total"
].sum().reset_index()

valid_orders = delivered.merge(order_totals, on="order_id")

rfm = valid_orders.groupby("customer_id").agg(
    frequency=("order_id", "count"),
    monetary=("line_total", "sum"),
    last_purchase=("order_date", "max")
).reset_index()

rfm = pd.DataFrame({"customer_id": customer_ids}).merge(
    rfm, on="customer_id", how="left"
)

rfm[["frequency", "monetary"]] = rfm[
    ["frequency", "monetary"]
].fillna(0)

rfm["recency"] = (
    pd.Timestamp("2026-10-30") -
    pd.to_datetime(rfm["last_purchase"])
).dt.days.fillna(999)

# 9. CUSTOMER SEGMENTATION
conditions = [
    rfm["recency"] > 180,
    rfm["recency"] > 90,
    (rfm["frequency"] >= 5) & (rfm["monetary"] >= 500),
    rfm["frequency"] >= 3,
    rfm["frequency"] <= 1
]

choices = [
    "Dormant", "At Risk", "Champions",
    "Loyal", "New"
]

rfm["segment"] = np.select(
    conditions, choices, default="Regular"
)

segment_names = [
    "Champions", "Loyal", "At Risk",
    "Dormant", "New", "Regular"
]

segment_rules = pd.DataFrame({
    "segment_id": [
        f"SEG-2026-{i:06d}" for i in range(1, 7)
    ],
    "segment_name": segment_names,
    "rule_type": ["RFM"] * 6
})

memberships = rfm[["customer_id", "segment"]].copy()

segment_map = dict(zip(
    segment_names, segment_rules["segment_id"]
))

memberships["segment_id"] = memberships["segment"].map(
    segment_map
)

memberships["membership_id"] = (
    "MEM-" + memberships["segment_id"] +
    "-" + memberships["customer_id"]
)

# 10. ACTION CATALOG
actions = pd.DataFrame({
    "action_id": [
        f"ACTN-2026-{i:06d}" for i in range(1, 7)
    ],
    "segment": segment_names,
    "action": [
        "VIP Reward", "Loyalty Points",
        "Win-back Offer", "Reactivation Offer",
        "Welcome Offer", "Product Suggestion"
    ]
})

# 11. RECOMMENDATIONS AND CONSENT CHECK
recommendations = memberships.merge(
    actions, on="segment"
).merge(
    consent[["customer_id", "marketing_allowed"]],
    on="customer_id"
)

recommendations["decision"] = np.where(
    recommendations["marketing_allowed"],
    "Eligible",
    "Suppressed_Consent"
)

recommendations["outcome_id"] = [
    f"OUT-{i:07d}"
    for i in range(1, len(recommendations)+1)
]

recommendations["outcome"] = np.where(
    recommendations["decision"] == "Eligible",
    "Generated",
    "Not_Sent"
)

# 12. LABELLED EDGE CASES
ground_truth = pd.DataFrame([
    ["EDGE-001", "Conflicting exclusive segments",
     "Flag customer for manual review"],
    ["EDGE-002", "Action violates consent",
     "Suppress contact"],
    ["EDGE-003", "Return not netted from spend",
     "Exclude returned order value"],
    ["EDGE-004", "Wrong action outcome attribution",
     "Reject mismatched action ID"]
], columns=[
    "issue_id", "issue_type", "expected_decision"
])

# 13. VALIDATION
checks = {
    "Unique customer IDs": customers.customer_id.is_unique,
    "Unique order IDs": orders.order_id.is_unique,
    "Unique product IDs": products.product_id.is_unique,
    "Valid customer references": orders.customer_id.isin(
        customer_ids
    ).all(),
    "Valid product references": order_lines.product_id.isin(
        product_ids
    ).all(),
    "Valid order references": order_lines.order_id.isin(
        order_ids
    ).all(),
    "Consent suppression": (
        recommendations.loc[
            ~recommendations["marketing_allowed"], "decision"
        ] == "Suppressed_Consent"
    ).all(),
    "No returned orders in RFM": not valid_orders["order_id"].isin(
        orders.loc[orders["status"] == "Returned", "order_id"]
    ).any()
}

# 14. EXPORT CSV FILES
tables = {
    "customers": customers,
    "products": products,
    "orders": orders,
    "order_lines": order_lines,
    "engagement_events": events,
    "support_cases": support,
    "consent": consent,
    "segment_rules": segment_rules,
    "segment_memberships": memberships,
    "action_catalog": actions,
    "action_outcomes": recommendations
}

for name, df in tables.items():
    df.to_csv(f"data/{name}.csv", index=False)

ground_truth.to_csv(
    "outputs/ground_truth.csv", index=False
)

# 15. VALIDATION REPORT
with open("outputs/validation_report.txt", "w") as file:
    for name, result in checks.items():
        file.write(
            f"{name}: {'PASS' if result else 'FAIL'}\n"
        )

print("\nYOTTAFLEX PROJECT COMPLETED")
print("Customers:", len(customers))
print("Orders:", len(orders))
print("Engagement events:", len(events))
print("Customer segments:", rfm["segment"].nunique())
print("Validation results:", checks)
print("Files saved in data/ and outputs/")
