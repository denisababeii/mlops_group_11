import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


try:
    from google.cloud import run_v2
except Exception:
    run_v2 = None


def get_backend_url() -> str:
    """
    Prefer local/dev backend via BACKEND env var.
    Otherwise try to discover Cloud Run service (if google-cloud-run is installed and auth is set up).
    """
    backend = os.environ.get("BACKEND")
    if backend:
        return backend.rstrip("/")

    if run_v2 is None:
        raise ValueError("BACKEND not set and google-cloud-run not available. Set BACKEND to use local API.")

    parent = "projects/mlops_group_11/locations/europe-west1"
    client = run_v2.ServicesClient()
    for service in client.list_services(parent=parent):
        if service.name.split("/")[-1] == "production-model":
            return service.uri.rstrip("/")

    raise ValueError("Backend service not found. Set BACKEND for local testing or deploy Cloud Run.")


def predict_poster(image: bytes, backend: str, threshold: float = 0.5, topk: int = 5) -> dict | None:
    """Send the image to the backend for prediction."""
    predict_url = f"{backend}/predict"
    params = {"threshold": threshold, "topk": topk}
    files={"file": image}

    response = requests.post(predict_url, params=params, files=files, timeout=10)
    if response.status_code == 200:
        return response.json()

    try:
        st.error(f"Backend error {response.status_code}: {response.json()}")
    except Exception:
        st.error(f"Backend error {response.status_code}: {response.text}")
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Movie Poster Genre Prediction 🎬")

    threshold = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.05)
    topk = st.slider("Top-K genres", 1, 10, 5, 1)

    uploaded_file = st.file_uploader("Upload a movie poster image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded poster", use_container_width=True)
            predict_clicked = st.button("Predict movie genre 🍿", use_container_width=True)

        if predict_clicked:
            with st.spinner("Predicting..."):
                result = predict_poster(image, backend=backend, threshold=threshold, topk=topk)

            if result is None:
                return

            st.subheader("🍿 Your movie is a...")
            predicted = result.get("predicted", [])
            if predicted:
                labels = [item["label"] for item in predicted]

                if len(labels) == 1:
                    genre_text = labels[0]
                elif len(labels) == 2:
                    genre_text = f"{labels[0]} and {labels[1]}"
                else:
                    genre_text = ", ".join(labels[:-1]) + f", and {labels[-1]}"

                st.write(f"{genre_text} movie!")
            else:
                st.write("Sadly, no genres exceeded the selected probability threshold.")


            st.subheader("Top-K predictions")
            topk_items = result.get("topk", [])
            if not topk_items:
                st.info("No predictions returned.")
                return

            # Table + bar chart for topk
            df_topk = pd.DataFrame(topk_items)
            st.dataframe(df_topk, use_container_width=True)

            chart_df = df_topk.set_index("label")[["probability"]]
            st.bar_chart(chart_df)

            st.subheader(f"All labels ≥ {result.get('threshold', threshold)}")
            predicted = result.get("predicted", [])
            if predicted:
                st.dataframe(pd.DataFrame(predicted), use_container_width=True)
            else:
                st.write("No labels above threshold.")


if __name__ == "__main__":
    main()
