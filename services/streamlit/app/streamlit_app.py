import streamlit as st
import pandas as pd
import requests
from pathlib import Path
import re

##############################################################
# Chemins

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"

API_URL = "http://api:8000"

##############################################################
# Bandeau lateral gauche

st.sidebar.title("MENU")

pages = [
    "Exploration",
    "Interprétabilité",
    "Modélisation",
    "Saisir un avis"
]

page = st.sidebar.radio("Aller vers", pages)

st.sidebar.image(
    IMAGES_DIR / "liora_logo.png",
    use_container_width=True
)

##############################################################
# Fonction appel API

def predict_review_api(review_text):

    response = requests.post(
        f"{API_URL}/predict",
        json={"text": review_text},
        headers={"x-api-key": "secret123"}
    )

    return response.json()

##############################################################
# Nettoyage texte

def clean_review_text(text: str):

    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

##############################################################
# Chargement dataset

@st.cache_data
def load_dataset():

    try:

        df = pd.read_csv(
            BASE_DIR / "test.csv",
            header=None,
            names=["label", "title", "text"]
        )

        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

        df_negative = df[df["label"] == 1]
        df_positive = df[df["label"] == 2]

        return df, df_negative, df_positive

    except Exception:

        return None, None, None


df, df_negative, df_positive = load_dataset()

#############################################################
# PAGE 0 - EXPLORATION
#############################################################

if page == pages[0]:

    st.title("Trustpilot Amazon Reviews")

    st.write("### Exploration")

    st.markdown("### 📊 Equilibre de la target")

    st.image(
        IMAGES_DIR / "repartition_sentiments.png",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📊 Longueur des avis")

    st.image(
        IMAGES_DIR / "boxplot_longueur_avis.png",
        use_container_width=True
    )

    st.divider()

    st.image(
        IMAGES_DIR / "boxplot_longueur_avis_par_sentiment.png",
        use_container_width=True
    )

    st.divider()

    st.image(
        IMAGES_DIR / "countplot_longueur_avis.png",
        use_container_width=True
    )

    st.divider()

    st.image(
        IMAGES_DIR / "violinplot_longueur_avis.png",
        caption="Distribution longueur des avis",
        use_container_width=True
    )

    st.divider()

    st.markdown("### ☁️ Nuages de mots")

    st.image(
        IMAGES_DIR / "wordcloud_positive.png",
        caption="Wordcloud – Positive reviews",
        use_container_width=True
    )

    st.divider()

    st.image(
        IMAGES_DIR / "wordcloud_negative.png",
        caption="Wordcloud – Negative reviews",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📊 Trigrammes")

    st.image(
        IMAGES_DIR / "barplot_trigrams_positive.png",
        caption="Top trigrams positive words",
        use_container_width=True
    )

    st.divider()

    st.image(
        IMAGES_DIR / "barplot_trigrams_negative.png",
        caption="Top trigrams negative words",
        use_container_width=True
    )

#############################################################
# PAGE 1 - INTERPRETABILITE
#############################################################

if page == pages[1]:

    st.title("Trustpilot Amazon Reviews")

    st.markdown("### 📐 SVM linéaire - coefficients TF-IDF")

    st.image(
        IMAGES_DIR / "interpretability_svm.png",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📐 Random Forest - sélection de variables (RFE)")

    st.image(
        IMAGES_DIR / "interpretability_random_forest.png",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📐 Interprétabilité XGBoost - Feature importance")

    st.image(
        IMAGES_DIR / "interpretability_xgboost.png",
        use_container_width=True
    )

    st.divider()

    st.markdown("### 📐 Interprétabilité DistillBERT - locale avec LIME")

    st.image(
        IMAGES_DIR / "interpretability_distilbert.png",
        use_container_width=True
    )

#############################################################
# PAGE 2 - MODELISATION
#############################################################

if page == pages[2]:

    st.title("Trustpilot Amazon Reviews")

    if df is None:

        st.warning("Dataset non disponible")

    else:

        st.markdown("### 🔮 Prédictions de quelques avis du dataset")

        random_state = 51

        def safe_sample(df, n, random_state):

            if len(df) == 0:
                return pd.DataFrame(columns=df.columns)

            return df.sample(
                n=min(n, len(df)),
                random_state=random_state
            )

        neg_samples = safe_sample(df_negative, 2, random_state)[["text", "label"]]
        pos_samples = safe_sample(df_positive, 1, random_state)[["text", "label"]]

        test_df = (
            pd.concat([neg_samples, pos_samples])
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

        label_mapping = {
            1: "Negative",
            2: "Positive"
        }

        for i, row in test_df.iterrows():

            review_text = row["text"]
            true_label = label_mapping[row["label"]]

            with st.spinner("Calling inference API..."):
                result = predict_review_api(review_text)

            st.markdown(f"### 📝 Avis {i+1}")

            st.info(review_text)

            #st.write(f"Sentiment réel (dataset) : **{true_label}**")

            st.caption("Inference powered by FastAPI")
            st.code(f"{API_URL}/predict")

            st.write("Cluster :", result["cluster_id"])
            #st.write("Theme :", result["theme"])
            st.write("Confidence :", round(result["confidence"], 3))

            with st.expander("API response (debug)"):
                st.json(result)

            st.divider()

#############################################################
# PAGE 3 - SAISIR UN AVIS
#############################################################

if page == pages[3]:

    st.title("Trustpilot Amazon Reviews")

    st.markdown("## ✍️ Saisissez un avis")

    review = st.text_area(
        label="(Ctrl+Entrée) pour valider",
        placeholder="Example: The movie stopped working after two weeks.",
        height=200
    )

    ICON_SIZE = 56

    THEME_ICONS = {
        "Product performance & value perception": "product_performance.png",
        "Entertainment and leisure products": "entertainment_leisure.png",
        "Movies, documentaries and audiovisual content": "movies_audiovisual.png",
        "Music albums & CDs": "music_albums.png",
        "Books & literature": "books_literature.png",
        "Technology & accessories": "technology_accessories.png",
    }

    if review.strip():

        try:

            st.caption("Inference powered by FastAPI")
            st.code(f"{API_URL}/predict")

            with st.spinner("Calling inference API..."):

                result = predict_review_api(review)

            st.markdown("### 🔮 Prediction")

            col_icon, col_text = st.columns([1,5], vertical_alignment="center")

            with col_icon:

                icon_file = THEME_ICONS.get(result["theme"])

                if icon_file:
                    st.image(IMAGES_DIR / icon_file, width=ICON_SIZE)

            with col_text:

                #st.markdown(
                #    f"<strong>Theme :</strong> {result['theme']}",
                #    unsafe_allow_html=True
                #)

                st.write("Cluster :", result["cluster_id"])
                st.write("Confidence :", round(result["confidence"],3))

            with st.expander("API response (debug)"):
                st.json(result)

        except Exception as e:

            st.error(f"API error: {e}")

    else:

        st.info("Veuillez saisir un avis pour lancer l'analyse.")
