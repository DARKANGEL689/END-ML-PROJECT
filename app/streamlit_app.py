"""
Streamlit Web Application for Substack Personalized Recommendation System.
Provides interactive reader recommendations, multi-angle explanations,
cold-start topic preference simulations, and full model benchmark leaderboards.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Setup base paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR
from src.explain import RecommendationExplainer, PERSONA_NAMES

st.set_page_config(
    page_title="Substack RecSys | ML Lab Project",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #ff6600; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
    .post-card { background-color: #fdfdfd; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .topic-pill { display: inline-block; background-color: #ffe8d6; color: #d9480f; padding: 3px 10px; border-radius: 12px; font-size: 0.82rem; font-weight: 600; }
    .reason-box { background-color: #f1f8ff; border-left: 4px solid #0366d6; padding: 8px 12px; margin-top: 8px; font-size: 0.9rem; color: #24292e; border-radius: 4px; }
    .metric-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 12px; text-align: center; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model_path = MODELS_DIR / "hybrid_recommender.joblib"
    posts_path = PROCESSED_DATA_DIR / "posts.csv"
    inter_path = PROCESSED_DATA_DIR / "interactions.csv"

    if not model_path.exists() or not posts_path.exists():
        return None, None, None

    model = joblib.load(model_path)
    df_posts = pd.read_csv(posts_path)
    df_inter = pd.read_csv(inter_path)
    return model, df_posts, df_inter


model, df_posts, df_interactions = load_artifacts()

# App Header
st.markdown('<div class="main-header">📚 Substack Content Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Personalized Newsletter Recommendation System using Machine Learning (All 13 Labs Integrated)</div>', unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ Trained model artifacts not found in `results/models/`.")
    st.info("Please execute the end-to-end pipeline first by running: `python run_all.py` in your terminal.")
    if st.button("🚀 Train Model Now Directly"):
        with st.spinner("Training all 13 algorithms and building artifacts..."):
            import run_all
            run_all.run_pipeline()
            st.success("Training complete! Please reload the page.")
            st.rerun()
    st.stop()

explainer = RecommendationExplainer(model, df_posts)

# Sidebar
st.sidebar.header("⚙️ Reader & Model Controls")
mode = st.sidebar.radio("Navigation Mode", ["Registered Reader (Warm Start)", "New Reader (Cold Start Simulation)", "Model Leaderboard & Analytics"])

all_users = list(model.all_users_)[:300]
selected_user = st.sidebar.selectbox("Select Reader ID", all_users, index=0)
top_k = st.sidebar.slider("Number of Recommendations (K)", min_value=3, max_value=20, value=10)
mmr_lambda = st.sidebar.slider("MMR Diversity Balance (Lambda)", min_value=0.1, max_value=1.0, value=0.7, step=0.05,
                               help="1.0 = Pure relevance ranking, 0.1 = Maximum diversity")

model.mmr_lambda = mmr_lambda

# -------------------------------------------------------------
# Mode 1: Registered Reader (Warm Start)
# -------------------------------------------------------------
if mode == "Registered Reader (Warm Start)":
    user_history = model.user_histories_.get(selected_user, [])
    persona_idx = model.user_personas_.get(selected_user, 0)
    persona_name = PERSONA_NAMES.get(persona_idx, f"Persona Group {persona_idx}")

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.markdown(f'<div class="metric-card"><b>Active Reader ID</b><br><span style="font-size:1.2rem; color:#ff6600;">{selected_user}</span></div>', unsafe_allow_html=True)
    with col_meta2:
        st.markdown(f'<div class="metric-card"><b>Assigned Persona Cohort</b><br><span style="font-size:1.1rem; color:#0366d6;">{persona_name}</span></div>', unsafe_allow_html=True)
    with col_meta3:
        st.markdown(f'<div class="metric-card"><b>Past Read Articles</b><br><span style="font-size:1.2rem; color:#28a745;">{len(user_history)} Posts</span></div>', unsafe_allow_html=True)

    st.write("")
    tab1, tab2 = st.tabs(["🎯 Top Recommendations & Explanations", "📜 Past Reading History"])

    with tab1:
        st.subheader(f"Top {top_k} Recommended Posts for {selected_user}")
        recs = model.recommend(selected_user, k=top_k)

        for rank, (item_id, score) in enumerate(recs, 1):
            exp = explainer.explain(selected_user, item_id)
            post_row = df_posts[df_posts["item_id"] == item_id].iloc[0] if len(df_posts[df_posts["item_id"] == item_id]) > 0 else None

            title = exp["title"]
            topic = exp["topic"]
            abstract = post_row["abstract"] if post_row is not None else ""
            pub_id = post_row["publication_id"] if post_row is not None else "pub_01"

            st.markdown(f"""
            <div class="post-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="topic-pill">{topic}</span>
                    <span style="font-size:0.85rem; color:#888;">Publication: <b>{pub_id}</b> | Match Score: <b>{score:.3f}</b></span>
                </div>
                <h4 style="margin: 8px 0 4px 0; color:#1a1a1a;">#{rank}. {title}</h4>
                <p style="font-size:0.92rem; color:#444; margin-bottom:8px;">{abstract[:220]}...</p>
                <div class="reason-box">
                    💡 <b>Why this is recommended:</b> {exp['primary_reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Reader's Recent Interaction History")
        if user_history:
            history_posts = df_posts[df_posts["item_id"].isin(user_history[:15])]
            st.dataframe(
                history_posts[["item_id", "topic", "title", "publication_id", "word_count"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recorded history for this user.")

# -------------------------------------------------------------
# Mode 2: New Reader Cold-Start Simulation
# -------------------------------------------------------------
elif mode == "New Reader (Cold Start Simulation)":
    st.subheader("🌱 Cold-Start Recommendation Simulation")
    st.write("Simulate onboarding a brand-new Substack reader with zero historical reading data. Select their initial topic interests:")

    all_topics = model.feature_pipeline.topic_list_
    selected_topics = st.multiselect("Choose Favorite Topics", all_topics, default=["Technology", "Finance"])

    if st.button("Generate Cold-Start Recommendations"):
        cold_recs = model.recommend("cold_start_guest", k=top_k, preferred_topics=selected_topics)
        st.success(f"Generated {len(cold_recs)} personalized recommendations tailored to: {', '.join(selected_topics)}")

        for rank, (item_id, score) in enumerate(cold_recs, 1):
            post_row = df_posts[df_posts["item_id"] == item_id].iloc[0] if len(df_posts[df_posts["item_id"] == item_id]) > 0 else None
            title = post_row["title"] if post_row is not None else item_id
            topic = post_row["topic"] if post_row is not None else "General"
            abstract = post_row["abstract"] if post_row is not None else ""

            st.markdown(f"""
            <div class="post-card">
                <span class="topic-pill">{topic}</span>
                <h4 style="margin: 6px 0;">#{rank}. {title}</h4>
                <p style="font-size:0.92rem; color:#444;">{abstract[:200]}...</p>
                <div class="reason-box">
                    🌱 <b>Cold-Start Routing:</b> Recommended because you selected interest in <b>{topic}</b> (routed through topic-specialized cluster centroid).
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Mode 3: Model Leaderboard & Analytics
# -------------------------------------------------------------
elif mode == "Model Leaderboard & Analytics":
    st.subheader("📊 Empirical Model Benchmark Leaderboard")
    
    comp_file = TABLES_DIR / "final_comparison.csv"
    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark table not yet generated. Run `python run_all.py` to populate.")

    st.subheader("🧪 Ablation Study: Component Impact on NDCG@10")
    ablation_file = TABLES_DIR / "ablation_study.csv"
    if ablation_file.exists():
        df_abl = pd.read_csv(ablation_file)
        st.dataframe(df_abl, use_container_width=True, hide_index=True)

    st.subheader("🖼️ Exploratory Figures & Diagnostic Visualizations")
    figures = list(FIGURES_DIR.glob("*.png"))
    if figures:
        selected_fig = st.selectbox("Select Diagnostic Figure", [f.name for f in figures])
        st.image(str(FIGURES_DIR / selected_fig), caption=selected_fig, use_container_width=True)
    else:
        st.info("Figures not found. Run `python run_all.py` to generate all 12 figures.")
