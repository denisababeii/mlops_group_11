# test_dummy_data.py
import io
from pathlib import Path

import requests
from PIL import Image

# Create a simple dummy image (100x100 red square)
img = Image.new("RGB", (100, 100), color="red")
img_bytes = io.BytesIO()
img.save(img_bytes, format="PNG")
img_bytes.seek(0)

# Test the API
url = "http://127.0.0.1:8000/predict"
files = {"file": ("test_image.png", img_bytes, "image/png")}
params = {"threshold": 0.5, "topk": 5}

response = requests.post(url, files=files, params=params)
print(response.json())

# Check if CSV was created
if Path("prediction_database.csv").exists():
    print("\n✓ CSV created!")
    print(Path("prediction_database.csv").read_text())
