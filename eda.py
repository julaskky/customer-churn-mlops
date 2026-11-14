
# ============================================================
# CUSTOMER CHURN PREDICTION
# EXPLORATORY DATA ANALYSIS (EDA)
# Algorithm: Logistic Regression
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "eda_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)
pd.set_option("display.float_format", "{:.2f}".format)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n========== LOADING DATASET ==========\n")

df = pd.read_csv(DATA_PATH)

# Convert TotalCharges to numeric.
# Blank or invalid entries become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ------------------------------------------------------------
# 3. DATASET STRUCTURE
# ------------------------------------------------------------

print("\n========== DATASET INFORMATION ==========\n")

print(df.info())

print("\nFirst five records:")
print(df.head())


# ------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n========== NUMERICAL STATISTICS ==========\n")

print(df.describe())

print("\n========== CATEGORICAL STATISTICS ==========\n")

print(df.describe(include="object"))


# ------------------------------------------------------------
# 5. MISSING VALUES
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========\n")

missing = df.isnull().sum()

print(missing[missing > 0])

print("\nTotal missing values:", df.isnull().sum().sum())


# Investigate missing TotalCharges records
print("\nRecords with missing TotalCharges:")

print(
    df.loc[
        df["TotalCharges"].isnull(),
        [
            "customerID",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Churn"
        ]
    ]
)


# ------------------------------------------------------------
# 6. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n========== DUPLICATE ANALYSIS ==========\n")

print("Duplicate rows:", df.duplicated().sum())

print("Duplicate customer IDs:", df["customerID"].duplicated().sum())


# ------------------------------------------------------------
# 7. TARGET VARIABLE DISTRIBUTION
# ------------------------------------------------------------

print("\n========== CHURN DISTRIBUTION ==========\n")

print(df["Churn"].value_counts())

print("\nChurn percentages:")

print(
    df["Churn"].value_counts(normalize=True) * 100
)


# ------------------------------------------------------------
# 8. NUMERICAL FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

df[numerical_features].hist(
    bins=30,
    figsize=(12, 8),
    edgecolor="black"
)

plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "numerical_distributions.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 9. CHURN DISTRIBUTION VISUALISATION
# ------------------------------------------------------------

churn_counts = df["Churn"].value_counts()

plt.figure(figsize=(7, 5))

churn_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "churn_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 10. CHURN BY CONTRACT TYPE
# ------------------------------------------------------------

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\n========== CHURN BY CONTRACT TYPE (%) ==========\n")

print(contract_churn)


# ------------------------------------------------------------
# 11. CHURN BY PAYMENT METHOD
# ------------------------------------------------------------

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\n========== CHURN BY PAYMENT METHOD (%) ==========\n")

print(payment_churn)


# ------------------------------------------------------------
# 12. SAVE EDA RESULTS
# ------------------------------------------------------------

contract_churn.to_csv(
    OUTPUT_DIR / "churn_by_contract.csv"
)

payment_churn.to_csv(
    OUTPUT_DIR / "churn_by_payment_method.csv"
)


# ------------------------------------------------------------
# 13. COMPLETION
# ------------------------------------------------------------

print("\n========== EDA COMPLETED ==========\n")

print("EDA output directory:", OUTPUT_DIR)

print("Generated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)
