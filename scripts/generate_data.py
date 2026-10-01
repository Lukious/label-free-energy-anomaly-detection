"""Generate synthetic dataset -> data/*.csv"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_dataset, save_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    ds = generate_synthetic_dataset(seed=42)
    save_dataset(ds, os.path.abspath(DATA_DIR))
    print(f"Saved {len(ds)} buildings to {os.path.abspath(DATA_DIR)}/")


if __name__ == "__main__":
    main()
