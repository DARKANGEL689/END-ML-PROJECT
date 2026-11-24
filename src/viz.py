"""
Visualization module for Substack Recommendation System.
Generates all 12+ required figures for exploratory data analysis (Assignment 2)
and downstream model evaluations, with unified aesthetics and automatic file export.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from src.config import FIGURES_DIR, DPI, PLOT_STYLE

# Set consistent matplotlib styling
plt.style.use("default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "figure.autolayout": False
})

PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ACCENT_COLOR = "#2ca02c"
COLOR_PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22"]


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """
    Save matplotlib figure to results/figures/ with high DPI and tight layout.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".png"):
        filename += ".png"
    out_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =====================================================================
# Assignment 2 Required EDA Visualizations
# =====================================================================

def plot_topic_distribution(df_posts: pd.DataFrame, filename: str = "01_topic_distribution.png") -> Path:
    """Figure 1: Category / Topic distribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    topic_counts = df_posts["topic"].value_counts()
    
    bars = ax.bar(topic_counts.index, topic_counts.values, color=PRIMARY_COLOR, edgecolor="black", alpha=0.85)
    ax.set_title("Distribution of Published Posts across Topics", pad=12)
    ax.set_xlabel("Topic / Section")
    ax.set_ylabel("Number of Posts")
    ax.set_xticks(range(len(topic_counts.index)))
    ax.set_xticklabels(topic_counts.index, rotation=35, ha="right")
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + (yval * 0.01), f"{int(yval):,}", ha="center", va="bottom", fontsize=8)
        
    return save_fig(fig, filename)


def plot_reads_per_user(df_interactions: pd.DataFrame, filename: str = "02_reads_per_user.png") -> Path:
    """Figure 2: Reads per user (histogram on log scale)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    user_counts = df_interactions["user_id"].value_counts()
    
    ax.hist(user_counts.values, bins=35, color=PRIMARY_COLOR, edgecolor="black", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Reader Activity Distribution (Log Scale)", pad=12)
    ax.set_xlabel("Number of Interactions per Reader")
    ax.set_ylabel("Reader Count (Log Scale)")
    
    median_val = np.median(user_counts.values)
    ax.axvline(median_val, color="red", linestyle="--", linewidth=1.5, label=f"Median: {median_val:.1f}")
    ax.legend()
    
    return save_fig(fig, filename)


def plot_item_popularity_long_tail(df_interactions: pd.DataFrame, filename: str = "03_item_popularity_long_tail.png") -> Path:
    """Figure 3: Long-tail post popularity (log-log plot)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    item_counts = df_interactions["item_id"].value_counts().values
    ranks = np.arange(1, len(item_counts) + 1)
    
    ax.loglog(ranks, item_counts, marker="o", markersize=2, linestyle="-", color=SECONDARY_COLOR)
    ax.set_title("Post Popularity Long-Tail Distribution (Log-Log)", pad=12)
    ax.set_xlabel("Item Popularity Rank (Log)")
    ax.set_ylabel("Interaction Count (Log)")
    ax.fill_between(ranks[:int(len(ranks)*0.2)], item_counts[:int(len(ranks)*0.2)], alpha=0.2, color="orange", label="Top 20% Head Items")
    ax.legend()
    
    return save_fig(fig, filename)


def plot_article_length_distribution(df_posts: pd.DataFrame, filename: str = "04_article_length_distribution.png") -> Path:
    """Figure 4: Article length distribution (histogram + density line)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    lengths = df_posts["word_count"].dropna()
    
    counts, bins, _ = ax.hist(lengths, bins=30, density=True, color="#6baed6", edgecolor="black", alpha=0.7, label="Histogram")
    
    # Simple Gaussian KDE density estimation
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(lengths)
    x_grid = np.linspace(lengths.min(), lengths.max(), 200)
    ax.plot(x_grid, kde(x_grid), color="#08519c", linewidth=2.0, label="Estimated KDE")
    
    ax.set_title("Distribution of Post Length (Word Count)", pad=12)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.legend()
    
    return save_fig(fig, filename)


def plot_ctr_by_topic(df_posts: pd.DataFrame, df_interactions: pd.DataFrame, filename: str = "05_ctr_by_topic.png") -> Path:
    """Figure 5: Click-through rate (CTR) by topic."""
    merged = df_interactions.merge(df_posts[["item_id", "topic"]], on="item_id", how="inner")
    topic_ctr = merged.groupby("topic")["clicked"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(topic_ctr.index, topic_ctr["mean"] * 100.0, color=ACCENT_COLOR, edgecolor="black", alpha=0.85)
    
    ax.set_title("Click-Through Rate (CTR %) across Topics", pad=12)
    ax.set_xlabel("Topic")
    ax.set_ylabel("CTR (%)")
    ax.set_xticks(range(len(topic_ctr.index)))
    ax.set_xticklabels(topic_ctr.index, rotation=35, ha="right")
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
        
    return save_fig(fig, filename)


def plot_engagement_by_hour(df_interactions: pd.DataFrame, filename: str = "06_engagement_by_hour.png") -> Path:
    """Figure 6: Reader engagement over the 24 hours of the day."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hours = df_interactions["timestamp"].dt.hour
    hourly_counts = hours.value_counts().sort_index()
    
    # Ensure all 24 hours exist
    hourly_full = pd.Series(0, index=range(24))
    hourly_full.update(hourly_counts)
    
    ax.plot(hourly_full.index, hourly_full.values, marker="s", color=PRIMARY_COLOR, linewidth=2, markersize=5)
    ax.fill_between(hourly_full.index, hourly_full.values, alpha=0.15, color=PRIMARY_COLOR)
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Reader Interaction Volume by Hour of Day", pad=12)
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Interaction Volume")
    
    return save_fig(fig, filename)


def plot_user_item_sparsity(sparse_mat: csr_matrix, sample_size: int = 250, filename: str = "07_user_item_sparsity.png") -> Path:
    """Figure 7: User-item matrix sparsity visualization (spy plot sample)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    sub_mat = sparse_mat[:min(sample_size, sparse_mat.shape[0]), :min(sample_size, sparse_mat.shape[1])]
    dense_sub = sub_mat.toarray()
    
    ax.spy(dense_sub, markersize=1.5, color="#2b5c8f")
    ax.set_title(f"User-Item Interaction Matrix Sparsity Sample ({sample_size}x{sample_size})", pad=12)
    ax.set_xlabel("Post Index")
    ax.set_ylabel("Reader Index")
    
    return save_fig(fig, filename)


def plot_top_terms_by_topic(df_posts: pd.DataFrame, top_n: int = 8, filename: str = "08_top_terms_by_topic.png") -> Path:
    """Figure 8: Top lexical terms per topic (bar chart grid alternative to word clouds)."""
    topics = df_posts["topic"].unique()[:6]  # Display up to 6 topics for readability
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    from sklearn.feature_extraction.text import CountVectorizer
    for idx, topic in enumerate(topics):
        ax = axes[idx]
        topic_text = df_posts[df_posts["topic"] == topic]["combined_text"].dropna()
        if len(topic_text) == 0:
            continue
        vec = CountVectorizer(max_features=top_n, stop_words="english")
        try:
            X = vec.fit_transform(topic_text)
            counts = np.asarray(X.sum(axis=0)).flatten()
            words = vec.get_feature_names_out()
            sorted_idx = np.argsort(counts)
            
            ax.barh(np.array(words)[sorted_idx], counts[sorted_idx], color=COLOR_PALETTE[idx % len(COLOR_PALETTE)], alpha=0.85)
            ax.set_title(f"Topic: {topic}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Frequency")
        except ValueError:
            pass
            
    return save_fig(fig, filename)


def plot_feature_distribution_grid(df_features: pd.DataFrame, filename: str = "09_feature_distribution_grid.png") -> Path:
    """Figure 9: Subplot grid comparing distributions of 4 key engineered features."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    cols = [
        ("word_count", "Word Count", "#3182bd"),
        ("read_time", "Read Time (s)", "#31a354"),
        ("recency_hours", "Recency (Hours)", "#fd8d3c"),
        ("title_length", "Title Length (Chars)", "#756bb1")
    ]
    
    for idx, (col, label, color) in enumerate(cols):
        ax = axes[idx]
        if col in df_features.columns:
            data = df_features[col].dropna()
            ax.hist(data, bins=25, color=color, edgecolor="black", alpha=0.75)
            ax.set_title(f"Distribution of {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Frequency")
            
    return save_fig(fig, filename)


def plot_feature_scatter_by_class(
    df_features: pd.DataFrame,
    x_col: str = "word_count",
    y_col: str = "read_time",
    label_col: str = "clicked",
    filename: str = "10_feature_scatter_by_class.png"
) -> Path:
    """Figure 10: Scatter plot of two key features coloured by engagement label."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    sample_df = df_features.sample(n=min(1000, len(df_features)), random_state=42)
    pos = sample_df[sample_df[label_col] == 1]
    neg = sample_df[sample_df[label_col] == 0]
    
    ax.scatter(neg[x_col], neg[y_col], c="gray", alpha=0.35, s=20, label="Unclicked (0)")
    ax.scatter(pos[x_col], pos[y_col], c="#e6550d", alpha=0.7, s=25, label="Clicked (1)")
    
    ax.set_title(f"Engagement Scatter: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}", pad=12)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.legend()
    
    return save_fig(fig, filename)


def plot_read_time_boxplots(df_interactions: pd.DataFrame, df_posts: pd.DataFrame, filename: str = "11_read_time_boxplots.png") -> Path:
    """Figure 11: Box plots of reading time across different content topics."""
    fig, ax = plt.subplots(figsize=(9, 5))
    merged = df_interactions.merge(df_posts[["item_id", "topic"]], on="item_id", how="inner")
    
    # Filter for clicked interactions and cap read time at 99th percentile for clean boxplot
    clicked_data = merged[merged["clicked"] == 1]
    q99 = clicked_data["read_time"].quantile(0.99)
    clicked_data = clicked_data[clicked_data["read_time"] <= q99]
    
    topics = clicked_data["topic"].value_counts().index
    data_by_topic = [clicked_data[clicked_data["topic"] == t]["read_time"].values for t in topics]
    
    bplot = ax.boxplot(data_by_topic, patch_artist=True, labels=topics, vert=True)
    
    for patch, color in zip(bplot["boxes"], COLOR_PALETTE * 2):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    ax.set_title("Reader Dwell Time by Post Topic (Clicked Interactions)", pad=12)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Read Time (Seconds)")
    ax.set_xticklabels(topics, rotation=35, ha="right")
    
    return save_fig(fig, filename)


def plot_correlation_heatmap(corr_df: pd.DataFrame, filename: str = "12_correlation_heatmap.png") -> Path:
    """Figure 12: Correlation matrix heatmap with numerical cell annotations."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    
    cax = ax.matshow(corr_df.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="left")
    ax.set_yticklabels(corr_df.columns)
    
    # Text annotations inside cells
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
            
    ax.set_title("Feature Correlation Matrix (Pearson)", pad=20)
    
    return save_fig(fig, filename)
