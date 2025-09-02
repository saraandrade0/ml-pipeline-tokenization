import random
import csv

# Normal transactions (label = 0)
normal_texts = [
    "Purchase made at local supermarket",
    "Utility bill payment",
    "Deposit into checking account",
    "Authorized ATM withdrawal",
    "Automatic payment for streaming subscription",
    "Salary payment credited to account",
    "Mobile recharge via app",
    "Restaurant bill paid with credit card",
    "Car loan installment payment",
    "Payment for monthly gym membership",
    "Transfer between own accounts",
    "Payment of internet service provider bill",
    "Insurance premium auto-debit",
    "Credit card limit increase request",
    "Online booking at hotel"
]

# Fraudulent transactions (label = 1)
fraud_texts = [
    "Customer reported a suspicious high-value transaction",
    "Unrecognized international transfer",
    "Login attempt from unknown device",
    "Online purchase from high-risk website",
    "Bank transfer to unregistered account",
    "Large withdrawal flagged by fraud system",
    "Suspicious overseas login attempt",
    "E-commerce purchase with stolen card",
    "ATM withdrawal at unusual location",
    "Multiple failed login attempts detected",
    "Unusual transfer during nighttime hours",
    "Attempted purchase from blacklisted website",
    "Suspicious transfer to new beneficiary",
    "Phishing-related unauthorized charge",
    "Unauthorized change of account details"
]

def generate_samples(n=100, output_file="data/samples.csv"):
    """
    Generate a synthetic dataset of normal (0) and fraudulent (1) transactions.
    """
    rows = []
    for i in range(1, n + 1):
        if random.random() < 0.5:
            text = random.choice(normal_texts)
            label = 0
        else:
            text = random.choice(fraud_texts)
            label = 1
        rows.append([i, text, label])

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        writer.writerows(rows)

    print(f"Generated {n} samples in {output_file}")

if __name__ == "__main__":
    # Generate 100 samples by default
    generate_samples(100)
