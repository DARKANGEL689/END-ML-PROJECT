from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import RANDOM_SEED

class StandardScalerScratch:

    def __init__(self, with_mean: bool=True, with_std: bool=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'StandardScalerScratch':
        X = np.asarray(X, dtype=np.float64)
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        if self.with_std:
            std = np.std(X, axis=0)
            self.scale_ = np.where(std == 0, 1.0, std)
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

class MinMaxScalerScratch:

    def __init__(self, feature_range: Tuple[float, float]=(0.0, 1.0)):
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'MinMaxScalerScratch':
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        rng = self.data_max_ - self.data_min_
        self.data_range_ = np.where(rng == 0, 1.0, rng)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        normed = (X - self.data_min_) / self.data_range_
        return normed * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    word = word.rstrip('e')
    vowels = 'aeiouy'
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and (not prev_is_vowel):
            count += 1
        prev_is_vowel = is_vowel
    return max(1, count)

def compute_flesch_reading_ease(text: str) -> float:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 60.0
    words = [w for w in text.split() if w.isalpha()]
    if len(words) == 0:
        return 60.0
    sentences = [s for s in text.replace('!', '.').replace('?', '.').split('.') if len(s.strip()) > 0]
    num_sentences = max(1, len(sentences))
    num_words = len(words)
    num_syllables = sum((count_syllables(w) for w in words))
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return float(np.clip(score, 0.0, 100.0))

class SubstackFeaturePipeline:

    def __init__(self, max_tfidf_features: int=100):
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english')
        self.scaler = StandardScalerScratch()
        self.is_fitted = False
        self.item_popularity_: Dict[str, float] = {}
        self.pub_popularity_: Dict[str, float] = {}
        self.user_topic_affinity_: Dict[str, Dict[str, float]] = {}
        self.user_avg_read_time_: Dict[str, float] = {}
        self.global_avg_read_time_: float = 60.0
        self.topic_list_: List[str] = []

    def fit(self, df_posts: pd.DataFrame, df_interactions: pd.DataFrame) -> 'SubstackFeaturePipeline':
        posts_clean = df_posts.copy()
        posts_clean['combined'] = posts_clean['title'].fillna('') + ' ' + posts_clean['abstract'].fillna('')
        self.tfidf_vectorizer.fit(posts_clean['combined'])
        self.topic_list_ = sorted(df_posts['topic'].dropna().unique().tolist())
        item_clicks = df_interactions.groupby('item_id')['clicked'].agg(['count', 'mean'])
        self.item_popularity_ = item_clicks['count'].to_dict()
        merged = df_interactions.merge(df_posts[['item_id', 'publication_id']], on='item_id', how='left')
        self.pub_popularity_ = merged.groupby('publication_id')['clicked'].count().to_dict()
        self.global_avg_read_time_ = float(df_interactions['read_time'].mean())
        self.user_avg_read_time_ = df_interactions.groupby('user_id')['read_time'].mean().to_dict()
        clicked_inter = df_interactions[df_interactions['clicked'] == 1].merge(df_posts[['item_id', 'topic']], on='item_id', how='left')
        user_topic_counts = clicked_inter.groupby(['user_id', 'topic']).size().unstack(fill_value=0)
        row_sums = user_topic_counts.sum(axis=1)
        row_sums = row_sums.replace(0, 1)
        user_affinity_df = user_topic_counts.div(row_sums, axis=0)
        self.user_topic_affinity_ = user_affinity_df.to_dict(orient='index')
        self.is_fitted = True
        return self

    def extract_features(self, df_posts: pd.DataFrame, df_interactions: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError('Pipeline must be fitted before transforming.')
        merged = df_interactions.merge(df_posts, on='item_id', how='left')
        title_lens = merged['title'].fillna('').apply(len)
        abstract_lens = merged['abstract'].fillna('').apply(len)
        word_counts = merged['word_count'].fillna(300).astype(float)
        readability = merged['abstract'].fillna('').apply(compute_flesch_reading_ease)
        ref_time = merged['timestamp'].max()
        if pd.api.types.is_datetime64_any_dtype(merged['timestamp']):
            post_time = merged['published_at'] if 'published_at' in merged.columns else merged['timestamp']
            age_hours = (ref_time - post_time).dt.total_seconds() / 3600.0
            age_hours = age_hours.clip(lower=0.1)
        else:
            age_hours = pd.Series(24.0, index=merged.index)
        tau = 72.0
        recency_decay = np.exp(-age_hours / tau)
        item_pop = merged['item_id'].map(lambda x: self.item_popularity_.get(x, 0)).astype(float)
        pub_pop = merged['publication_id'].map(lambda x: self.pub_popularity_.get(x, 0)).astype(float)
        user_avg_rt = merged['user_id'].map(lambda u: self.user_avg_read_time_.get(u, self.global_avg_read_time_)).astype(float)
        topic_match_scores = []
        is_top_topic = []
        for _, row in merged.iterrows():
            u = row['user_id']
            t = row['topic']
            u_dict = self.user_topic_affinity_.get(u, {})
            score = u_dict.get(t, 0.0)
            topic_match_scores.append(score)
            if len(u_dict) > 0:
                top_t = max(u_dict, key=u_dict.get)
                is_top_topic.append(1 if t == top_t else 0)
            else:
                is_top_topic.append(0)
        features_df = pd.DataFrame({'user_id': merged['user_id'], 'item_id': merged['item_id'], 'clicked': merged['clicked'].astype(int), 'read_time': merged['read_time'].astype(float), 'word_count': word_counts, 'title_length': title_lens, 'abstract_length': abstract_lens, 'readability_score': readability, 'recency_hours': age_hours, 'recency_decay': recency_decay, 'item_popularity': np.log1p(item_pop), 'publication_popularity': np.log1p(pub_pop), 'user_avg_read_time': user_avg_rt, 'topic_affinity_score': np.array(topic_match_scores), 'is_top_topic': np.array(is_top_topic), 'topic': merged['topic'].fillna('General')})
        return features_df

def discretize_features_for_id3(df_features: pd.DataFrame) -> pd.DataFrame:
    df_discrete = pd.DataFrame()
    df_discrete['clicked'] = df_features['clicked'].astype(int)
    df_discrete['length_bucket'] = pd.cut(df_features['word_count'], bins=[-np.inf, 800, 1800, np.inf], labels=['Short', 'Medium', 'Long'])
    df_discrete['recency_bucket'] = pd.cut(df_features['recency_hours'], bins=[-np.inf, 24, 72, np.inf], labels=['Fresh', 'Recent', 'Older'])
    df_discrete['popularity_bucket'] = pd.qcut(df_features['item_popularity'].rank(method='first'), q=3, labels=['Low', 'Medium', 'High'])
    df_discrete['topic_affinity_bucket'] = pd.cut(df_features['topic_affinity_score'], bins=[-np.inf, 0.1, 0.35, np.inf], labels=['Low', 'Medium', 'High'])
    df_discrete['topic_match'] = df_features['is_top_topic'].map({1: 'Yes', 0: 'No'})
    df_discrete['topic'] = df_features['topic'].astype(str)
    return df_discrete

def clip_outliers_iqr(series: pd.Series, factor: float=1.5) -> pd.Series:
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr
    return series.clip(lower=lower_bound, upper=upper_bound)
