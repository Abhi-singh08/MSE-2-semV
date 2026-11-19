from .data import load_training_frame
from .config import DATA_PROCESSED

def main():
    df = load_training_frame()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "train_FD001_processed.csv"
    df.to_csv(out, index=False)
    print(f"Prepared NASA FD001 training data: {len(df):,} rows, {df.engine_id.nunique()} engines")
    print(f"Saved to: {out}")

if __name__ == "__main__":
    main()
