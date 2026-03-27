from xgboost import XGBRegressor

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import Set


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keyword_threshold: int = 40):
        self.keyword_threshold = keyword_threshold

        self.genres_set: Set[str] = set()
        self.directors_set: Set[str] = set()
        self.locations_set: Set[str] = set()
        self.keywords_set: Set[str] = set()

        self.gender_cols = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']

    def fit(self, X: pd.DataFrame, y=None):

        if 'genres' in X:
            self.genres_set = {g for row in X['genres'] for g in row}

        if 'directors' in X:
            self.directors_set = {d for row in X['directors'] for d in row}

        if 'filming_locations' in X:
            self.locations_set = {loc for row in X['filming_locations'] for loc in row}

        if 'keywords' in X:
            from collections import Counter

            flat = [k for row in X['keywords'] for k in row]
            counts = Counter(flat)

            self.keywords_set = {
                k for k, v in counts.items()
                if v >= self.keyword_threshold
            }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X = X.copy()

        for col in self.gender_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')

        if 'genres' in X:
            X['n_genres'] = X['genres'].apply(len)

        if 'directors' in X:
            X['n_directors'] = X['directors'].apply(len)

        if 'filming_locations' in X:
            X['n_locations'] = X['filming_locations'].apply(len)

        if 'keywords' in X:
            X['n_keywords'] = X['keywords'].apply(len)

        X['actors_known_sum'] = (
            X['actor_0_known_movies'] +
            X['actor_1_known_movies'] +
            X['actor_2_known_movies']
        )

        new_cols = {}

        def encode(column, values, prefix):
            if column not in X.columns:
                return
            for val in values:
                new_cols[f"{prefix}_{val}"] = [
                    1 if val in row else 0
                    for row in X[column]
                ]

        encode('genres', self.genres_set, 'genre')
        encode('directors', self.directors_set, 'dir')
        encode('filming_locations', self.locations_set, 'loc')
        encode('keywords', self.keywords_set, 'kw')

        encoded = pd.DataFrame(new_cols, index=X.index)

        X = X.drop(columns=['genres', 'directors', 'filming_locations', 'keywords'],
                   errors='ignore')

        X = pd.concat([X, encoded], axis=1)

        return X


def train_model_and_predict(train_file: str, test_file: str) -> np.ndarray:

    train_df = pd.read_json(train_file, lines=True)
    test_df = pd.read_json(test_file, lines=True)

    y = train_df["awards"]
    X = train_df.drop(columns=["awards"])

    params = {
        "n_estimators": 500,
        "learning_rate": 0.06,
        "max_depth": 4,
        "min_child_weight": 4,
        "gamma": 0.0007,
        "subsample": 0.72,
        "colsample_bytree": 0.87,
        "reg_alpha": 0.01,
        "enable_categorical": True,
        "random_state": 42
    }

    pipe = Pipeline([
        ("features", FeatureTransformer(keyword_threshold=40)),
        ("model", XGBRegressor(**params))
    ])

    pipe.fit(X, y)

    preds = pipe.predict(test_df).astype(np.float64)

    return preds
