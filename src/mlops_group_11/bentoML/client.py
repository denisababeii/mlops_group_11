import numpy as np
from service import PosterClassifierService

# Initialize service
service = PosterClassifierService()

# Example movie poster URL
image_url = "https://m.media-amazon.com/images/M/MV5BNTMwMTk0Y2QtY2VhNy00OGYwLThkMjMtZjkwMGI3MTJiMjAyXkEyXkFqcGc@._V1_.jpg"

try:
    result = service.predict_from_url(image_url=image_url, threshold=0.5)
    print(f'  Image URL: {image_url}')
    print(f'  Predicted genres: {result["genres"]}')
    print(f'  Top 5 probabilities:')
    top_5_indices = np.argsort(result["probabilities"])[-5:][::-1]
    for idx in top_5_indices:
        print(f'    {service.genre_labels[idx]}: {result["probabilities"][idx]:.4f}')
except Exception as e:
    print(f'  Error: {e}')
