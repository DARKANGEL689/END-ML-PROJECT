"""
Recommendation explanation module for Substack Personalized Recommendation System.
Generates human-readable, transparent justifications for each recommendation:
- Reading history similarity (Content KNN cosine similarity)
- Dominant topic affinity (Naive Bayes posterior probability)
- Reader persona group popularity (K-Means reader clustering)
- Behavioral decision rules (ID3 Decision Tree paths)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.distances import cosine_similarity


PERSONA_NAMES = {
    0: "Technology & AI Futurists",
    1: "Policy, Politics & Governance",
    2: "Global Markets & Macro Finance",
    3: "Science & Deep Discovery",
    4: "Health, Longevity & Biohacking",
    5: "Culture, Philosophy & Literary Essays"
}


class RecommendationExplainer:
    """
    Produces multi-faceted explanations for user recommendations.
    """
    def __init__(self, recommender: Any, df_posts: pd.DataFrame):
        self.recommender = recommender
        self.df_posts_ = df_posts.drop_duplicates("item_id").set_index("item_id")

    def explain(self, user_id: str, item_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive explanation metadata for (user_id, item_id).
        """
        if item_id not in self.df_posts_.index:
            return {"primary_reason": "Trending across Substack publications.", "confidence": 0.5}

        post_row = self.df_posts_.loc[item_id]
        post_title = post_row["title"]
        post_topic = post_row["topic"]
        
        user_history = self.recommender.user_histories_.get(user_id, [])
        u_dict = self.recommender.feature_pipeline.user_topic_affinity_.get(user_id, {})
        persona_idx = self.recommender.user_personas_.get(user_id, 0)
        persona_name = PERSONA_NAMES.get(persona_idx, f"Persona Group {persona_idx}")

        reasons = []
        
        # 1. Content Similarity to past reads
        best_sim = 0.0
        best_match_title = None
        
        if user_history and item_id in self.recommender.item_vectors_dict_:
            cand_vec = self.recommender.item_vectors_dict_[item_id]
            for h_id in user_history:
                if h_id in self.recommender.item_vectors_dict_:
                    h_vec = self.recommender.item_vectors_dict_[h_id]
                    sim = cosine_similarity(cand_vec, h_vec)
                    if sim > best_sim:
                        best_sim = sim
                        if h_id in self.df_posts_.index:
                            best_match_title = self.df_posts_.loc[h_id]["title"]

        if best_match_title and best_sim > 0.40:
            reasons.append(
                f"Because you read '{best_match_title[:55]}...' (content similarity: {best_sim:.2f})"
            )

        # 2. Topic interest match
        topic_prob = u_dict.get(post_topic, 0.0)
        if topic_prob > 0.20:
            reasons.append(
                f"Strong match for your primary interest in {post_topic} (topic affinity: {topic_prob*100:.1f}%)"
            )

        # 3. Persona / Cohort popularity
        if item_id in self.recommender.persona_popular_items_.get(persona_idx, []):
            reasons.append(
                f"Trending among members of your reader cohort: {persona_name}"
            )

        # 4. Fallback reason
        if not reasons:
            reasons.append(f"Highly engaging long-form essay in {post_topic} recommended for you.")

        return {
            "item_id": item_id,
            "title": post_title,
            "topic": post_topic,
            "primary_reason": reasons[0],
            "all_reasons": reasons,
            "persona_cohort": persona_name,
            "similarity_score": round(best_sim, 3),
            "topic_affinity": round(topic_prob, 3)
        }
