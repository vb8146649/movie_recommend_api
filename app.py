from fastapi import FastAPI, Query, HTTPException
from typing import List
import pickle
import pandas as pd
import numpy as np
import gdown
import os

app = FastAPI()

# Download if not already present
def download_if_missing(file_id, output):
    if not os.path.exists(output):
        gdown.download(id=file_id, output=output, quiet=False)

download_if_missing("1RC8J9gRnSi-QprNglycFlZq0c4WS53Ra", "similarity.pkl")
download_if_missing("1JHyLK143WH7TGYhO0hrdVDbBg5xUtqL_", "movies_dict.pkl")

# Load pickled data
with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

with open("movies_dict.pkl", "rb") as f:
    movies = pd.DataFrame(pickle.load(f))  # must have 'id' and 'title'

# Recommend function based on movie ID
def recommend_by_id(movie_id: int, history_ids: List[int], top_k: int = 20):
    if movie_id not in movies['id'].values:
        raise ValueError("Movie ID not found.")

    if movie_id not in history_ids:
        history_ids.append(movie_id)
    history_ids = history_ids[-5:]

    distances = np.zeros(len(similarity[0]))

    for hid in history_ids:
        match = movies[movies['id'] == hid]
        if not match.empty:
            idx = match.index[0]
            d = 3 if hid == movie_id else 1
            distances += similarity[idx] * d
    distances /= len(history_ids)

    recommended_ids = [
        int(movies.iloc[i[0]].id)
        for i in sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[:top_k]
    ]
    return recommended_ids

# API endpoint
@app.get("/recommend")
def get_recommendations(movie_id: int = Query(...), history: List[int] = Query([])):
    try:
        recommended_ids = recommend_by_id(movie_id, history)
        return {"movie_ids": recommended_ids}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
