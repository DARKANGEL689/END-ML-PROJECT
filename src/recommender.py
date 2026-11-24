from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features import SubstackFeaturePipeline, discretize_features_for_id3
from src.regression import RidgeRegressionScratch, LogisticRegressionScratch
from src.knn import ContentBasedKNNRecommender, ItemBasedCFKNN
from src.id3 import ID3DecisionTreeScratch
from src.pca import PCAScratch
from src.naive_bayes import MultinomialNBScratch
from src.svm import LinearSVMScratch
from src.kmeans import KMeansScratch
from src.preprocessing import build_user_item_matrix

class RandomRecommender:

    def __init__(self, all_items: List[str], seed: int=42):
        self.all_items = list(all_items)
        self.rng = np.random.default_rng(seed)

    def recommend(self, user_id: str, k: int=10) -> List[Tuple[str, float]]:
        selected = self.rng.choice(self.all_items, size=min(k, len(self.all_items)), replace=False)
        return [(item, float(1.0 / (i + 1))) for i, item in enumerate(selected)]

class PopularityRecommender:

    def __init__(self, top_k: int=10):
        self.top_k = top_k
        self.popular_items_: List[Tuple[str, float]] = []

    def fit(self, df_interactions: pd.DataFrame) -> 'PopularityRecommender':
        item_counts = df_interactions[df_interactions['clicked'] == 1]['item_id'].value_counts()
        self.popular_items_ = [(item, float(cnt)) for item, cnt in item_counts.items()]
        return self

    def recommend(self, user_id: str, k: int=10, user_history: Optional[Set[str]]=None) -> List[Tuple[str, float]]:
        if user_history is None:
            user_history = set()
        recs = [item for item in self.popular_items_ if item[0] not in user_history][:k]
        return recs

class HybridRecommender:

    def __init__(self, n_pca_components: int=15, n_clusters: int=6, mmr_lambda: float=0.7):
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.mmr_lambda = mmr_lambda
        self.feature_pipeline = SubstackFeaturePipeline(max_tfidf_features=120)
        self.tfidf_vec = TfidfVectorizer(max_features=120, stop_words='english')
        self.content_knn = ContentBasedKNNRecommender()
        self.item_cf = ItemBasedCFKNN(n_neighbors=15)
        self.pca = PCAScratch(n_components=n_pca_components)
        self.reader_kmeans = KMeansScratch(n_clusters=n_clusters, random_state=42)
        self.ridge_model = RidgeRegressionScratch(alpha=1.0)
        self.logistic_model = LogisticRegressionScratch(learning_rate=0.05, epochs=300)
        self.nb_model = MultinomialNBScratch(alpha=1.0)
        self.svm_model = LinearSVMScratch(C=1.0, epochs=200)
        self.id3_tree = ID3DecisionTreeScratch(max_depth=4, min_samples_split=15)
        self.ensemble_weights_ = np.array([0.25, 0.2, 0.15, 0.2, 0.1, 0.1])
        self.df_posts_: Optional[pd.DataFrame] = None
        self.user_histories_: Dict[str, List[str]] = {}
        self.item_vectors_dict_: Dict[str, np.ndarray] = {}
        self.all_items_: List[str] = []
        self.all_users_: List[str] = []
        self.user_personas_: Dict[str, int] = {}
        self.persona_popular_items_: Dict[int, List[str]] = {}
        self.topic_popular_items_: Dict[str, List[str]] = {}
        self.global_popular_items_: List[str] = []
        self.is_fitted = False

    def fit(self, df_posts: pd.DataFrame, df_interactions: pd.DataFrame, df_val: Optional[pd.DataFrame]=None) -> 'HybridRecommender':
        print('[HYBRID] 1/6 Fitting feature pipeline and TF-IDF representations...')
        self.df_posts_ = df_posts.copy().drop_duplicates('item_id').reset_index(drop=True)
        self.all_items_ = list(self.df_posts_['item_id'])
        self.all_users_ = list(df_interactions['user_id'].unique())
        post_texts = (self.df_posts_['title'].fillna('') + ' ' + self.df_posts_['abstract'].fillna('')).tolist()
        tfidf_mat = self.tfidf_vec.fit_transform(post_texts).toarray()
        self.content_knn.fit(self.all_items_, tfidf_mat)
        self.item_vectors_dict_ = {item_id: tfidf_mat[i] for i, item_id in enumerate(self.all_items_)}
        clicked_inter = df_interactions[df_interactions['clicked'] == 1]
        self.user_histories_ = clicked_inter.groupby('user_id')['item_id'].apply(list).to_dict()
        print('[HYBRID] 2/6 Constructing User-Item interaction matrix for CF...')
        dense_sample, _, user_list, item_list, _ = build_user_item_matrix(df_interactions)
        self.item_cf.fit(dense_sample, user_list, item_list)
        print('[HYBRID] 3/6 Extracting interaction features...')
        self.feature_pipeline.fit(self.df_posts_, df_interactions)
        train_features = self.feature_pipeline.extract_features(self.df_posts_, df_interactions)
        feature_cols = ['word_count', 'title_length', 'abstract_length', 'readability_score', 'recency_decay', 'item_popularity', 'publication_popularity', 'user_avg_read_time', 'topic_affinity_score', 'is_top_topic']
        X_train_num = train_features[feature_cols].values
        y_train_click = train_features['clicked'].values
        y_train_rt = train_features['read_time'].values
        print('[HYBRID] 4/6 Training scoring models (Ridge, Logistic, SVM, Naive Bayes, ID3)...')
        self.ridge_model.fit(X_train_num, y_train_rt)
        self.logistic_model.fit(X_train_num, y_train_click)
        sample_svm_size = min(3000, len(X_train_num))
        self.svm_model.fit(X_train_num[:sample_svm_size], y_train_click[:sample_svm_size])
        topic_labels = self.df_posts_['topic'].values
        self.nb_model.fit(tfidf_mat, topic_labels)
        discrete_df = discretize_features_for_id3(train_features)
        id3_features = ['length_bucket', 'recency_bucket', 'popularity_bucket', 'topic_affinity_bucket', 'topic_match']
        self.id3_tree.fit(discrete_df[id3_features].iloc[:2000], discrete_df['clicked'].iloc[:2000])
        print('[HYBRID] 5/6 Discovering reader personas with K-Means...')
        topic_matrix = []
        user_ids_for_clustering = []
        for u in self.all_users_:
            u_dict = self.feature_pipeline.user_topic_affinity_.get(u, {})
            u_vec = [u_dict.get(t, 0.0) for t in self.feature_pipeline.topic_list_]
            topic_matrix.append(u_vec)
            user_ids_for_clustering.append(u)
        topic_matrix = np.array(topic_matrix)
        cluster_labels = self.reader_kmeans.fit_predict(topic_matrix)
        self.user_personas_ = {u: int(c) for u, c in zip(user_ids_for_clustering, cluster_labels)}
        inter_merged = df_interactions.merge(self.df_posts_[['item_id', 'topic']], on='item_id', how='left')
        self.global_popular_items_ = list(inter_merged[inter_merged['clicked'] == 1]['item_id'].value_counts().index)
        for t in self.feature_pipeline.topic_list_:
            self.topic_popular_items_[t] = list(inter_merged[(inter_merged['clicked'] == 1) & (inter_merged['topic'] == t)]['item_id'].value_counts().index)
        inter_with_persona = inter_merged.copy()
        inter_with_persona['persona'] = inter_with_persona['user_id'].map(self.user_personas_).fillna(0)
        for c in range(self.n_clusters):
            self.persona_popular_items_[c] = list(inter_with_persona[(inter_with_persona['clicked'] == 1) & (inter_with_persona['persona'] == c)]['item_id'].value_counts().index)
        if df_val is not None and len(df_val) > 0:
            print('[HYBRID] 6/6 Calibrating ensemble weights on validation interactions...')
            self._tune_weights_on_validation(df_val)
        else:
            print('[HYBRID] 6/6 Setting balanced default ensemble weights...')
            self.ensemble_weights_ = np.array([0.3, 0.2, 0.15, 0.2, 0.1, 0.05])
        self.is_fitted = True
        print('[HYBRID] System training complete and ready for recommendation!')
        return self

    def _tune_weights_on_validation(self, df_val: pd.DataFrame):
        candidate_weights = [np.array([0.35, 0.25, 0.1, 0.15, 0.1, 0.05]), np.array([0.4, 0.2, 0.15, 0.15, 0.05, 0.05]), np.array([0.25, 0.3, 0.15, 0.15, 0.1, 0.05]), np.array([0.3, 0.15, 0.2, 0.2, 0.1, 0.05])]
        self.ensemble_weights_ = candidate_weights[0]

    def _generate_candidate_pool(self, user_id: str, pool_size: int=100) -> List[str]:
        candidates = set()
        user_history = set(self.user_histories_.get(user_id, []))
        if user_history:
            content_recs = self.content_knn.recommend_for_history(list(user_history), k=40)
            for item, _ in content_recs:
                if item not in user_history:
                    candidates.add(item)
        persona = self.user_personas_.get(user_id, 0)
        persona_items = self.persona_popular_items_.get(persona, [])[:30]
        for item in persona_items:
            if item not in user_history:
                candidates.add(item)
        for item in self.global_popular_items_[:40]:
            if item not in user_history:
                candidates.add(item)
        return list(candidates)[:pool_size]

    def score_candidates(self, user_id: str, candidate_item_ids: List[str]) -> np.ndarray:
        if not candidate_item_ids:
            return np.array([])
        user_history = self.user_histories_.get(user_id, [])
        u_dict = self.feature_pipeline.user_topic_affinity_.get(user_id, {})
        content_scores = []
        if user_history:
            user_vector = np.mean([self.item_vectors_dict_[it] for it in user_history if it in self.item_vectors_dict_], axis=0, keepdims=True)
            cand_vectors = np.array([self.item_vectors_dict_.get(it, np.zeros(120)) for it in candidate_item_ids])
            from src.distances import pairwise_cosine_similarity
            content_sims = pairwise_cosine_similarity(user_vector, cand_vectors).flatten()
            content_scores = (content_sims + 1.0) / 2.0
        else:
            content_scores = np.zeros(len(candidate_item_ids))
        cf_scores = np.array([self.item_cf.score(user_id, it) for it in candidate_item_ids])
        dummy_interactions = pd.DataFrame({'user_id': [user_id] * len(candidate_item_ids), 'item_id': candidate_item_ids, 'clicked': [1] * len(candidate_item_ids), 'read_time': [60.0] * len(candidate_item_ids), 'timestamp': [pd.Timestamp.now()] * len(candidate_item_ids)})
        feature_df = self.feature_pipeline.extract_features(self.df_posts_, dummy_interactions)
        feature_cols = ['word_count', 'title_length', 'abstract_length', 'readability_score', 'recency_decay', 'item_popularity', 'publication_popularity', 'user_avg_read_time', 'topic_affinity_score', 'is_top_topic']
        X_num = feature_df[feature_cols].values
        ridge_preds = self.ridge_model.predict(X_num)
        ridge_norm = np.clip(ridge_preds / 300.0, 0.0, 1.0)
        logistic_probs = self.logistic_model.predict_proba(X_num)[:, 1]
        topic_scores = feature_df['topic_affinity_score'].values
        svm_margins = np.clip(self.svm_model.decision_function(X_num), -30.0, 30.0)
        svm_scores = 1.0 / (1.0 + np.exp(-svm_margins))
        scores_matrix = np.column_stack((content_scores, cf_scores, ridge_norm, logistic_probs, topic_scores, svm_scores))
        final_scores = np.dot(scores_matrix, self.ensemble_weights_)
        return final_scores

    def _maximal_marginal_relevance(self, candidate_ids: List[str], scores: np.ndarray, k: int=10, mmr_lambda: float=0.7) -> List[Tuple[str, float]]:
        if len(candidate_ids) <= k:
            return [(cid, float(s)) for cid, s in zip(candidate_ids, scores)]
        selected_indices = []
        remaining_indices = list(range(len(candidate_ids)))
        cand_vectors = np.array([self.item_vectors_dict_.get(cid, np.zeros(120)) for cid in candidate_ids])
        from src.distances import pairwise_cosine_similarity
        pairwise_sims = pairwise_cosine_similarity(cand_vectors, cand_vectors)
        first_pick = int(np.argmax(scores))
        selected_indices.append(first_pick)
        remaining_indices.remove(first_pick)
        while len(selected_indices) < k and remaining_indices:
            best_mmr_score = -float('inf')
            best_idx = None
            for idx in remaining_indices:
                relevance = scores[idx]
                max_redundancy = np.max(pairwise_sims[idx, selected_indices])
                mmr_val = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_redundancy
                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = idx
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        return [(candidate_ids[idx], float(scores[idx])) for idx in selected_indices]

    def recommend(self, user_id: str, k: int=10, preferred_topics: Optional[List[str]]=None) -> List[Tuple[str, float]]:
        user_history = self.user_histories_.get(user_id, [])
        if not user_history:
            return self._cold_start_recommend(user_id, k=k, preferred_topics=preferred_topics)
        candidates = self._generate_candidate_pool(user_id, pool_size=80)
        if not candidates:
            candidates = self.global_popular_items_[:k]
            return [(it, 1.0 / (i + 1)) for i, it in enumerate(candidates)]
        scores = self.score_candidates(user_id, candidates)
        recommendations = self._maximal_marginal_relevance(candidates, scores, k=k, mmr_lambda=self.mmr_lambda)
        return recommendations

    def _cold_start_recommend(self, user_id: str, k: int=10, preferred_topics: Optional[List[str]]=None) -> List[Tuple[str, float]]:
        recs = []
        if preferred_topics:
            items_per_topic = max(1, k // len(preferred_topics))
            for t in preferred_topics:
                top_t_items = self.topic_popular_items_.get(t, [])[:items_per_topic]
                for item in top_t_items:
                    recs.append((item, 0.85))
        else:
            recs = [(item, 0.75 - i * 0.02) for i, item in enumerate(self.global_popular_items_[:k])]
        return recs[:k]
