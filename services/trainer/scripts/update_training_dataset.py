from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SAS_PATH = PROJECT_ROOT / "data/sas/trustpilot_new_reviews.csv"
TRAIN_PATH = PROJECT_ROOT / "data/raw/train.csv"
ARCHIVE_PATH = PROJECT_ROOT / "data/archive/integrated_reviews.csv"

THRESHOLD = 50

def main():

    sas_df = pd.read_csv(SAS_PATH)
    nb_reviews = len(sas_df)

    print("Nombre de reviews dans le SAS :", nb_reviews)

    if nb_reviews < THRESHOLD:
        print("Seuil non atteint → on ne fait rien")
        return

    print("Seuil atteint → mise à jour du dataset")

    train_df = pd.read_csv(TRAIN_PATH)
    print("Nombre de reviews dans train.csv :", len(train_df))

    updated_train_df = pd.concat([train_df, sas_df], ignore_index=True)
    print("Nombre de reviews après fusion :", len(updated_train_df))

    # mise à jour du train dataset
    updated_train_df.to_csv(TRAIN_PATH, index=False)
    print("train.csv mis à jour")

    #achive des reviews
    archive_df = pd.read_csv(ARCHIVE_PATH)
    updated_archive_df = pd.concat([archive_df, sas_df], ignore_index=True)
    updated_archive_df.to_csv(ARCHIVE_PATH, index=False)
    print("reviews archivées")

    #ider le SAS (en gardant le header)
    sas_df.iloc[0:0].to_csv(SAS_PATH, index=False)
    print("SAS vidé")



if __name__ == "__main__":
    main()
