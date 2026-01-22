import os

import pandas as pd
import requests
import streamlit as st


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    backend = os.environ.get("BACKEND_URL")
    if backend is None:
        st.error("BACKEND_URL environment variable not set")
        raise ValueError("BACKEND_URL not configured")
    return backend


def predict_poster(image: bytes, backend: str, threshold: float = 0.5, topk: int = 5) -> dict | None:
    """Send the image to the backend for prediction."""
    predict_url = f"{backend}/predict"
    params = {"threshold": threshold, "topk": topk}
    files = {"file": image}

    response = requests.post(predict_url, params=params, files=files, timeout=60)
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
