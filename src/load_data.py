"""
Load the raw sample into SQLite and build the modelling table with SQL.

    from src.load_data import build_database, load_model_table
    build_database()              # data/lc_sample.csv -> data/lending_club.db
    df = load_model_table()       # SELECT * FROM loans_model
"""
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
RAW_CSV = DATA_DIR / "lc_sample.csv"
DB_PATH = DATA_DIR / "lending_club.db"

TRAIN_YEARS = range(2007, 2015)   # 2007-2014: development sample
TEST_YEAR = 2015                  # out-of-time validation sample


def _read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def build_database(csv_path: Path = RAW_CSV, db_path: Path = DB_PATH) -> None:
    """Load the raw CSV into table raw_loans, then run sql/features.sql."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run prepare_sample.py first (see README)."
        )
    raw = pd.read_csv(csv_path, low_memory=False)
    with sqlite3.connect(db_path) as con:
        raw.to_sql("raw_loans", con, if_exists="replace", index=False)
        con.executescript(_read_sql("features.sql"))
        n = con.execute("SELECT COUNT(*) FROM loans_model").fetchone()[0]
    print(f"raw_loans: {len(raw):,} rows -> loans_model: {n:,} rows ({db_path.name})")


def run_query(sql_file: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Run one of the .sql files in sql/ and return the result."""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(_read_sql(sql_file), con)


def load_model_table(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not Path(db_path).exists():
        build_database(db_path=db_path)
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM loans_model", con)


def split_out_of_time(df: pd.DataFrame):
    """Development = 2007-2014 vintages, out-of-time test = 2015 vintage."""
    train = df[df["issue_year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test = df[df["issue_year"] == TEST_YEAR].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    build_database()
    df = load_model_table()
    train, test = split_out_of_time(df)
    print(f"train {len(train):,} (default {train.is_default.mean():.1%}) | "
          f"test {len(test):,} (default {test.is_default.mean():.1%})")
