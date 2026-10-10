"""
Step 0: shrink the Lending Club accepted-loans file to what the project needs.

Usage (run on your own machine, from the folder with the Kaggle file):
    pip install pandas
    python prepare_sample.py accepted_2007_to_2018Q4.csv.gz

Works with the .csv or .csv.gz version. Output: lc_sample.csv (upload this).
"""
import sys
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "accepted_2007_to_2018Q4.csv.gz"
OUT = "lc_sample.csv"
N_SAMPLE = 200_000
SEED = 42

# Application-time features (known at decision time) + outcome/profit fields
COLS = [
    # identifiers / timing
    "id", "issue_d",
    # loan terms
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "purpose",
    # borrower profile
    "emp_length", "home_ownership", "annual_inc", "verification_status", "addr_state",
    "dti", "application_type",
    # bureau attributes
    "fico_range_low", "fico_range_high", "earliest_cr_line", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
    # outcome + post-outcome cash flows (profit simulation ONLY, never features)
    "loan_status", "total_pymnt", "recoveries", "collection_recovery_fee",
]

TERMINAL = {"Fully Paid", "Charged Off"}

chunks = []
reader = pd.read_csv(SRC, usecols=COLS, chunksize=250_000, low_memory=False)
for i, chunk in enumerate(reader, 1):
    chunk = chunk[chunk["loan_status"].isin(TERMINAL)]
    chunks.append(chunk)
    print(f"chunk {i}: kept {len(chunk):,} completed loans")

df = pd.concat(chunks, ignore_index=True)
print(f"total completed loans: {len(df):,}")

if len(df) > N_SAMPLE:
    df = df.sample(n=N_SAMPLE, random_state=SEED)

df.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(df):,} rows, default rate "
      f"{(df['loan_status'] == 'Charged Off').mean():.1%}")
