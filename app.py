import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(BASE_DIR / "vectorizer_tfidf.pkl")
    iso = joblib.load(BASE_DIR / "model_isolation_forest.pkl")
    clf = joblib.load(BASE_DIR / "model_location_classifier.pkl")
    df = pd.read_csv(BASE_DIR / "news_with_flags.csv")
    return tfidf, iso, clf, df


tfidf, iso, clf, df = load_artifacts()

st.markdown(
    """
    <div style='background-color:#e8f5e9; padding:15px; border-radius:10px;'>
        <h2 style='color:#1b5e20;'>Hyperlocal News Anomaly & Source Discrepancy Detector</h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <ul style='font-size:17px;'>
        <li>🔴 <b style='color:#c62828;'>Detect anomalous articles</b></li>
        <li>🔵 <b style='color:#1565c0;'>Check location mismatches between metadata & content</b></li>
    </ul>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🔎 Suspicious Articles Explorer", "📝 Analyse New Article"])

# ---------- Tab 1: Suspicious Articles Explorer ----------
with tab1:
    st.subheader("Detected Suspicious Articles")

    suspicious = df[df["is_suspicious"] == True].copy()

    st.write(f"Total suspicious articles: **{len(suspicious)}**")

    cols_to_show = [
        "Heading",
        "Article",
        "Location",
        "location_extracted",
        "location_predicted",
        "anomaly_score",
        "is_suspicious",
    ]

    st.dataframe(suspicious[cols_to_show].head(100))

# ---------- Tab 2: Analyse New Article ----------
with tab2:
    st.subheader("Check a New Article")

    heading_input = st.text_input("Article Heading")
    article_input = st.text_area("Full Article Text", height=200)
    meta_location = st.text_input(
        "Reported Location (optional, e.g., 'US', 'Pakistan', 'India')",
        ""
    )

    if st.button("Analyse"):
        text = (heading_input + " " + article_input).strip()

        if len(text) < 20:
            st.warning("Please enter a proper heading + article text.")
        else:
            X_new = tfidf.transform([text])

            anomaly_label = iso.predict(X_new)[0]
            anomaly_score = float(-iso.decision_function(X_new)[0])

            try:
                loc_pred = clf.predict(X_new)[0]
            except Exception:
                loc_pred = "UNKNOWN"

            st.write("---")
            st.write(f"**Predicted Location from Content:** `{loc_pred}`")
            st.write(f"**Anomaly Score:** `{anomaly_score:.4f}`")

            if anomaly_label == -1:
                st.error("⚠ This article looks **anomalous** compared to normal news.")
            else:
                st.success("✅ This article seems normal compared to the training data.")

            if meta_location.strip():
                mismatch = (meta_location.strip().lower() != loc_pred.lower())
                if mismatch:
                    st.warning(
                        f"⚠ Reported location `{meta_location}` "
                        f"does NOT match predicted location `{loc_pred}`."
                    )
                else:
                    st.info(
                        f"✅ Reported location `{meta_location}` "
                        f"matches the predicted location."
                    )


