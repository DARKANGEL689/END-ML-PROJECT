"""
Data loader module for Substack Personalized Recommendation System.
Loads MIND dataset (news proxy) or generates realistic synthetic Substack interactions,
along with 20 Newsgroups, secondary text datasets, and tabular benchmarks.
"""

from typing import Tuple, Dict, Any, Optional
import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups, load_iris, load_wine, load_breast_cancer

from src.config import (
    MIND_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SAMPLE_USERS,
    SAMPLE_ITEMS,
    SYNTHETIC_CONFIG,
    DATA_SOURCE
)


def generate_synthetic_substack_data(
    num_posts: int = SYNTHETIC_CONFIG["num_posts"],
    num_topics: int = SYNTHETIC_CONFIG["num_topics"],
    num_publications: int = SYNTHETIC_CONFIG["num_publications"],
    num_readers: int = SYNTHETIC_CONFIG["num_readers"],
    target_interactions: int = SYNTHETIC_CONFIG["target_interactions"],
    output_dir: Path = PROCESSED_DATA_DIR,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a high-fidelity synthetic Substack proxy dataset when offline.
    
    Generates posts across distinct topics, assigned to publications,
    and creates readers with latent Dirichlet topic preferences, reading times,
    and power-law interaction frequencies.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_posts, df_interactions)
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Define Topics and Vocabulary Lexicons
    topics = [
        "Technology", "Politics", "Finance", "Science", "Health",
        "Culture", "Philosophy", "Design", "Writing", "Productivity"
    ]
    topic_vocab = {
        "Technology": ["AI", "algorithm", "software", "GPU", "transformer", "neural", "compiler", "cloud", "silicon", "agent", "code", "latency", "distributed", "API"],
        "Politics": ["policy", "election", "democracy", "senate", "voters", "diplomacy", "treaty", "campaign", "parliament", "regulation", "bill", "governance"],
        "Finance": ["market", "inflation", "equities", "portfolio", "treasury", "bonds", "yield", "capital", "recession", "valuation", "liquidity", "dividend"],
        "Science": ["physics", "quantum", "genome", "crispr", "astronomy", "telescope", "particle", "entropy", "evolution", "chemistry", "neuroscience", "relativity"],
        "Health": ["longevity", "metabolism", "nutrition", "cardio", "sleep", "circadian", "dopamine", "cortisol", "strength", "cellular", "immunity", "fasting"],
        "Culture": ["cinema", "literature", "society", "art", "criticism", "modernity", "nostalgia", "urbanism", "music", "aesthetic", "canon", "memoir"],
        "Philosophy": ["epistemology", "ethics", "stoicism", "existentialism", "consciousness", "logic", "dialectic", "teleology", "ontology", "pragmatism", "morality"],
        "Design": ["typography", "interface", "minimalism", "ergonomics", "visual", "grid", "affordance", "layout", "aesthetic", "prototype", "hierarchy", "craft"],
        "Writing": ["prose", "narrative", "draft", "storytelling", "syntax", "voice", "essay", "revision", "rhetoric", "metaphor", "dialogue", "publishing"],
        "Productivity": ["habits", "deep work", "focus", "time blocking", "systems", "leverage", "attention", "workflows", "discipline", "execution", "goals"]
    }
    
    # 2. Generate Publications
    pub_names = [
        "Silicon Frontier", "The Political Ledger", "Compound Wisdom", "Quantum Dispatch",
        "Vitality Protocol", "Cultural Digest", "The Dialectic", "Craft & Surface",
        "The Craft of Prose", "Deep Focus Chronicle", "Alpha Vector", "Civic Forum",
        "Global Macro Letter", "Cosmos & Life", "Primal Health", "Modern Canvas",
        "Reflective Mind", "Design Architecture", "Literary Review", "Iterative Edge"
    ][:num_publications]
    
    pub_df = pd.DataFrame({
        "publication_id": [f"pub_{i:02d}" for i in range(num_publications)],
        "publication_name": pub_names,
        "primary_topic": [topics[i % len(topics)] for i in range(num_publications)],
        "follower_count": rng.integers(500, 50000, size=num_publications)
    })
    
    # 3. Generate Posts
    post_records = []
    base_timestamp = pd.Timestamp("2024-01-01 08:00:00")
    
    for i in range(num_posts):
        item_id = f"post_{i:05d}"
        topic = topics[i % len(topics)]
        # Pick a publication matching the topic or nearby
        matching_pubs = pub_df[pub_df["primary_topic"] == topic]["publication_id"].values
        pub_id = rng.choice(matching_pubs) if len(matching_pubs) > 0 else f"pub_{rng.integers(0, num_publications):02d}"
        author_id = f"author_{hashlib.md5(f'{pub_id}_{topic}'.encode()).hexdigest()[:6]}"
        
        words = topic_vocab[topic]
        w1, w2, w3 = rng.choice(words, size=3, replace=False)
        title = f"The Future of {w1.capitalize()} and {w2.capitalize()}: A Deep Dive into {w3.capitalize()}"
        
        # Realistic abstract / text body
        abstract_words = rng.choice(words, size=rng.integers(25, 60), replace=True)
        abstract = f"In this essay, we explore how {w1} impacts contemporary {topic.lower()}. " + \
                   " ".join(abstract_words) + f". We conclude with strategic implications for {w2} and {w3}."
        
        word_count = rng.integers(400, 3200)
        published_at = base_timestamp + pd.Timedelta(hours=int(rng.integers(0, 24 * 60)))
        
        post_records.append({
            "item_id": item_id,
            "topic": topic,
            "section": f"{topic} Analysis",
            "publication_id": pub_id,
            "author_id": author_id,
            "title": title,
            "abstract": abstract,
            "word_count": word_count,
            "published_at": published_at
        })
        
    df_posts = pd.DataFrame(post_records)
    
    # 4. Generate Readers with Latent Preferences
    # Readers have Dirichlet topic preference vector (alpha concentration controls specialization)
    alpha = np.ones(num_topics) * 0.5
    user_topic_pref = rng.dirichlet(alpha, size=num_readers)  # (num_readers, num_topics)
    reading_speeds = rng.normal(loc=230, scale=35, size=num_readers).clip(120, 380)  # words per minute
    
    # Power-law user activity distribution
    raw_activity = rng.pareto(a=1.8, size=num_readers) + 1.0
    activity_probs = raw_activity / raw_activity.sum()
    interactions_per_user = (activity_probs * target_interactions).astype(int) + 3
    
    # 5. Generate Interactions
    interaction_records = []
    topic_to_idx = {t: idx for idx, t in enumerate(topics)}
    post_topics_idx = df_posts["topic"].map(topic_to_idx).values
    post_item_ids = df_posts["item_id"].values
    post_word_counts = df_posts["word_count"].values
    
    for u_idx in range(num_readers):
        u_id = f"user_{u_idx:05d}"
        n_inter = min(interactions_per_user[u_idx], 120)
        u_pref = user_topic_pref[u_idx]
        
        # Probability of selecting post is proportional to topic alignment
        post_probs = u_pref[post_topics_idx] + 0.05
        post_probs /= post_probs.sum()
        
        selected_post_indices = rng.choice(len(df_posts), size=n_inter, replace=False, p=post_probs)
        
        for p_idx in selected_post_indices:
            p_id = post_item_ids[p_idx]
            t_idx = post_topics_idx[p_idx]
            match_score = u_pref[t_idx]
            
            # Click probability increases with topic match
            click_prob = 1.0 / (1.0 + np.exp(-4.0 * (match_score - 0.2)))
            clicked = int(rng.uniform() < click_prob)
            
            # Read time modeled conditionally on click and post length
            w_count = post_word_counts[p_idx]
            expected_read_sec = (w_count / reading_speeds[u_idx]) * 60.0
            
            if clicked == 1:
                completion_rate = rng.beta(a=3.5, b=1.5)
                read_time = float(np.round(expected_read_sec * completion_rate, 1))
                liked = int(read_time > (expected_read_sec * 0.6) and (rng.uniform() < 0.45))
                subscribed = int(liked == 1 and (rng.uniform() < 0.25))
            else:
                read_time = float(np.round(rng.uniform(2.0, 15.0), 1))
                liked = 0
                subscribed = 0
                
            interaction_time = base_timestamp + pd.Timedelta(hours=int(rng.integers(1, 24 * 90)))
            
            interaction_records.append({
                "user_id": u_id,
                "item_id": p_id,
                "clicked": clicked,
                "read_time": read_time,
                "liked": liked,
                "subscribed": subscribed,
                "timestamp": interaction_time
            })
            
    df_interactions = pd.DataFrame(interaction_records)
    
    # Save to processed directory
    df_posts.to_csv(output_dir / "posts.csv", index=False)
    df_interactions.to_csv(output_dir / "interactions.csv", index=False)
    pub_df.to_csv(output_dir / "publications.csv", index=False)
    
    print(f"[DATA] Generated synthetic dataset: {len(df_posts)} posts, {len(df_interactions)} interactions, {num_readers} readers.")
    return df_posts, df_interactions


def load_synthetic_substack_data(output_dir: Path = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached synthetic dataset or generate if not yet created."""
    posts_file = output_dir / "posts.csv"
    interactions_file = output_dir / "interactions.csv"
    
    if posts_file.exists() and interactions_file.exists():
        df_posts = pd.read_csv(posts_file, parse_dates=["published_at"])
        df_interactions = pd.read_csv(interactions_file, parse_dates=["timestamp"])
        return df_posts, df_interactions
    
    return generate_synthetic_substack_data(output_dir=output_dir)


def load_mind_dataset(sample_users: int = SAMPLE_USERS) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Load Microsoft News Dataset (MIND-small) mapped to Substack terminology.
    
    MIND files:
    - news.tsv: news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    - behaviors.tsv: impression_id, user_id, time, history, impressions
    """
    train_news_path = MIND_DIR / "train" / "news.tsv"
    train_behaviors_path = MIND_DIR / "train" / "behaviors.tsv"
    
    if not train_news_path.exists() or not train_behaviors_path.exists():
        return None

    try:
        print("[MIND] Parsing news.tsv...")
        news_cols = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        df_news = pd.read_csv(
            train_news_path,
            sep="\t",
            names=news_cols,
            usecols=["news_id", "category", "subcategory", "title", "abstract"],
            encoding="utf-8"
        )
        
        # Clean nulls
        df_news["title"] = df_news["title"].fillna("")
        df_news["abstract"] = df_news["abstract"].fillna("")
        df_news["category"] = df_news["category"].fillna("General")
        df_news["subcategory"] = df_news["subcategory"].fillna("General")
        
        # Create deterministic simulated publication_id and author_id based on subcategory
        def _get_pub_id(subcat: str) -> str:
            h = int(hashlib.md5(f"{subcat}_{RANDOM_SEED}".encode()).hexdigest(), 16)
            return f"pub_{(h % 50):02d}"

        def _get_author_id(subcat: str, news_id: str) -> str:
            h = int(hashlib.md5(f"{subcat}_{news_id[:4]}_{RANDOM_SEED}".encode()).hexdigest(), 16)
            return f"author_{(h % 150):03d}"

        df_news["publication_id"] = df_news["subcategory"].apply(_get_pub_id)
        df_news["author_id"] = [
            _get_author_id(s, n) for s, n in zip(df_news["subcategory"], df_news["news_id"])
        ]
        
        # Map to Substack schema
        df_posts = pd.DataFrame({
            "item_id": df_news["news_id"],
            "topic": df_news["category"],
            "section": df_news["subcategory"],
            "publication_id": df_news["publication_id"],
            "author_id": df_news["author_id"],
            "title": df_news["title"],
            "abstract": df_news["abstract"],
            "word_count": df_news["abstract"].apply(lambda s: len(str(s).split()) * 8 + 120),
            "published_at": pd.Timestamp("2024-01-01")
        })
        
        # Load behaviors
        print("[MIND] Parsing behaviors.tsv...")
        beh_cols = ["impression_id", "user_id", "time", "history", "impressions"]
        
        # Sample users to adhere to laptop constraints
        records = []
        user_count = 0
        target_users = set()
        
        with open(train_behaviors_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                u_id = parts[1]
                target_users.add(u_id)
                if len(target_users) >= sample_users:
                    break

        print(f"[MIND] Selected sample of {len(target_users)} users. Unpacking click logs...")
        rng = np.random.default_rng(RANDOM_SEED)
        
        with open(train_behaviors_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                u_id = parts[1]
                if u_id not in target_users:
                    continue
                time_str = parts[2]
                impressions = parts[4].split()
                
                # Unpack impressions: e.g., 'N12345-1' (clicked) or 'N12345-0' (not clicked)
                for imp in impressions:
                    if "-" not in imp:
                        continue
                    item_id, click_str = imp.split("-")
                    clicked = int(click_str)
                    
                    # Synthesize read_time and liked from click status
                    if clicked == 1:
                        read_time = float(np.round(rng.exponential(scale=180.0) + 45.0, 1))
                        liked = int(read_time > 150.0 and (rng.uniform() < 0.40))
                        subscribed = int(liked == 1 and (rng.uniform() < 0.20))
                    else:
                        read_time = float(np.round(rng.uniform(1.0, 15.0), 1))
                        liked = 0
                        subscribed = 0

                    records.append({
                        "user_id": u_id,
                        "item_id": item_id,
                        "clicked": clicked,
                        "read_time": read_time,
                        "liked": liked,
                        "subscribed": subscribed,
                        "timestamp": pd.to_datetime(time_str)
                    })

        df_interactions = pd.DataFrame(records)
        
        # Filter posts to those that appear in interactions or sample
        active_items = set(df_interactions["item_id"].unique())
        df_posts = df_posts[df_posts["item_id"].isin(active_items)].reset_index(drop=True)
        
        print(f"[MIND] Successfully loaded {len(df_posts)} items and {len(df_interactions)} interaction records.")
        return df_posts, df_interactions

    except Exception as e:
        print(f"[WARN] Error parsing MIND dataset: {e}. Falling back to synthetic.")
        return None


def load_data(
    source: str = DATA_SOURCE,
    sample_users: int = SAMPLE_USERS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main data loading gateway with automatic fallback to synthetic data.
    
    Args:
        source: "mind" | "synthetic"
        sample_users: number of users to sample if using MIND
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (df_posts, df_interactions)
    """
    if source.lower() == "mind":
        result = load_mind_dataset(sample_users=sample_users)
        if result is not None:
            return result
        print("[DATA] MIND dataset not available locally. Using synthetic dataset.")
        
    return load_synthetic_substack_data()


def load_20newsgroups(subset: str = "all") -> Tuple[list, np.ndarray, list]:
    """
    Load 20 Newsgroups text classification dataset.
    
    Returns:
        Tuple[list, np.ndarray, list]: (texts, targets, target_names)
    """
    try:
        bunch = fetch_20newsgroups(
            subset=subset,
            remove=("headers", "footers", "quotes"),
            random_state=RANDOM_SEED
        )
        texts = [t for t in bunch.data]
        targets = bunch.target
        target_names = bunch.target_names
        return texts, targets, target_names
    except Exception as e:
        print(f"[WARN] Could not fetch 20 Newsgroups ({e}). Generating fallback corpus.")
        rng = np.random.default_rng(RANDOM_SEED)
        target_names = [f"topic_{i}" for i in range(10)]
        texts = []
        targets = []
        for i in range(1000):
            t = rng.integers(0, 10)
            texts.append(f"Synthetic document number {i} discussing {target_names[t]} with diverse terminology.")
            targets.append(t)
        return texts, np.array(targets), target_names


def load_tabular_validation_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, list]]:
    """
    Load Iris, Wine, and Breast Cancer datasets for testing and scratch algorithm validation.
    """
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    
    return {
        "iris": (iris.data, iris.target, list(iris.feature_names)),
        "wine": (wine.data, wine.target, list(wine.feature_names)),
        "breast_cancer": (cancer.data, cancer.target, list(cancer.feature_names))
    }


def print_summary_table(df_posts: pd.DataFrame, df_interactions: pd.DataFrame) -> str:
    """
    Compute and print the dataset summary table (users, items, sparsity, balance).
    """
    n_users = df_interactions["user_id"].nunique()
    n_items = df_posts["item_id"].nunique()
    n_interactions = len(df_interactions)
    total_possible = n_users * n_items
    sparsity = (1.0 - (n_interactions / total_possible)) * 100.0 if total_possible > 0 else 0.0
    
    click_balance = df_interactions["clicked"].value_counts(normalize=True).to_dict()
    pos_rate = click_balance.get(1, 0.0) * 100.0
    neg_rate = click_balance.get(0, 0.0) * 100.0
    
    summary = f"""
======================================================================
               SUBSTACK RECOMMENDER DATASET SUMMARY
======================================================================
Total Posts (Items)          : {n_items:,}
Total Readers (Users)        : {n_users:,}
Total Interactions           : {n_interactions:,}
User-Item Matrix Sparsity    : {sparsity:.4f}%
Positive Clicks (1)          : {pos_rate:.2f}%
Unclicked Impressions (0)    : {neg_rate:.2f}%
Average Reads per User       : {n_interactions / n_users:.2f}
Average Reads per Post       : {n_interactions / n_items:.2f}
Number of Topics             : {df_posts['topic'].nunique()}
======================================================================
"""
    print(summary)
    return summary
