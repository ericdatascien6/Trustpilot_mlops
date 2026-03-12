from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_PATH = PROJECT_ROOT / "data/dataset_source/amazon_reviews.csv"
SAS_PATH = PROJECT_ROOT / "data/sas/trustpilot_new_reviews.csv"
OFFSET_PATH = PROJECT_ROOT / "data/metadata/stream_offset.txt"

BATCH_SIZE = 50


def read_offset():
    with open(OFFSET_PATH, "r") as f:
        return int(f.read().strip())


def write_offset(offset):
    with open(OFFSET_PATH, "w") as f:
        f.write(str(offset))


def main():

    offset = read_offset()

    df = pd.read_csv(
        SOURCE_PATH,
        skiprows=range(1, offset),
        nrows=BATCH_SIZE,
        header=None,
        names=["label","title", "text"]
    )

    # fin du dataset → reset
    if df.empty:
        print("Fin du dataset atteinte → reset offset")
        write_offset(0)
        return

    df.to_csv(SAS_PATH, mode="a", header=False, index=False)

    new_offset = offset + len(df)
    write_offset(new_offset)

    print(f"{len(df)} reviews ajoutées au SAS")
    print(f"Nouvel offset : {new_offset}")


if __name__ == "__main__":
    main()
