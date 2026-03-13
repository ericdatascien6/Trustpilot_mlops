from pathlib import Path
import pandas as pd
import json
from datetime import datetime

SAS_PATH = Path("/opt/airflow/data/sas/trustpilot_new_reviews.csv")
TRAIN_PATH = Path("/opt/airflow/data/raw/train.csv")
ARCHIVE_PATH = Path("/opt/airflow/data/archive/integrated_reviews.csv")
STATUS_PATH = Path("/opt/airflow/data/sas/ingestion_status.json")

THRESHOLD = 200


def write_status(threshold_reached, sas_count, moved_to_train):
    status = {
        "threshold_reached": threshold_reached,
        "sas_review_count": sas_count,
        "threshold": THRESHOLD,
        "moved_to_train": moved_to_train,
        "updated_at": datetime.utcnow().isoformat()
    }

    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

    print("ingestion_status.json mis à jour")


def main():

    if not SAS_PATH.exists():
        print("SAS file introuvable")

        write_status(
            threshold_reached=False,
            sas_count=0,
            moved_to_train=0
        )
        return

    sas_df = pd.read_csv(SAS_PATH)
    nb_reviews = len(sas_df)

    print("Nombre de reviews dans le SAS :", nb_reviews)

    if nb_reviews < THRESHOLD:
        print("Seuil non atteint → on ne fait rien")

        write_status(
            threshold_reached=False,
            sas_count=nb_reviews,
            moved_to_train=0
        )
        return

    print("Seuil atteint → mise à jour du dataset")

    train_df = pd.read_csv(TRAIN_PATH)
    print("Nombre de reviews dans train.csv :", len(train_df))

    updated_train_df = pd.concat([train_df, sas_df], ignore_index=True)
    print("Nombre de reviews après fusion :", len(updated_train_df))

    updated_train_df.to_csv(TRAIN_PATH, index=False)
    print("train.csv mis à jour")

    archive_df = pd.read_csv(ARCHIVE_PATH)
    updated_archive_df = pd.concat([archive_df, sas_df], ignore_index=True)
    updated_archive_df.to_csv(ARCHIVE_PATH, index=False)
    print("reviews archivées")

    sas_df.iloc[0:0].to_csv(SAS_PATH, index=False)
    print("SAS vidé")

    write_status(
        threshold_reached=True,
        sas_count=nb_reviews,
        moved_to_train=nb_reviews
    )


if __name__ == "__main__":
    main()
