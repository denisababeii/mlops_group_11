import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

from mlops_group_11.api.fast_api import app

client = TestClient(app)


def test_health_responds():
    """
    API test: verify /health endpoint responds correctly.
    """
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_with_dummy_model(monkeypatch):
    """
    API test: verify /predict endpoint works with a dummy model.
    """
    import mlops_group_11.api.fast_api as api

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # logits shape must match NUM_CLASSES=24
            return torch.zeros((x.shape[0], 24), dtype=torch.float32)

    def fake_ensure_loaded():
        api._device = torch.device("cpu")
        api._model = DummyModel()
        api._labels = [f"label_{i}" for i in range(24)]

    monkeypatch.setattr(api, "_ensure_loaded", fake_ensure_loaded)

    # Create a dummy image
    img = Image.new("RGB", (224, 224))
    buf = io.BytesIO()  # in-memory buffer
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf, "image/jpeg")}
    r = client.post("/predict?threshold=0.5&topk=5", files=files)

    assert r.status_code == 200
    data = r.json()
    assert "predicted" in data
    assert "topk" in data
    assert len(data["topk"]) == 5
