import os
from pathlib import Path

from locust import HttpUser, task, between

SAMPLE_IMAGE = Path(os.getenv("LOCUST_IMAGE", "tests/performancetests/sample_avengers.png"))


class PosterAPIUser(HttpUser):
    # Simulated "think time" between requests
    wait_time = between(0.2, 1.0)

    @task(3)
    def health(self):
        # Called ~3x more often than predict()
        self.client.get("/health")

    @task(1)
    def predict(self):
        # Upload an image to /predict
        if not SAMPLE_IMAGE.exists():
            return

        with SAMPLE_IMAGE.open("rb") as f:
            files = {"file": ("sample.jpg", f, "image/jpeg")}
            # Use query params like real users
            self.client.post("/predict?threshold=0.5&topk=5", files=files)
