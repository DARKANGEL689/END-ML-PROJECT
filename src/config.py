"""
Global configuration for Substack Personalized Recommendation System.
Centralizes paths, hyperparameters, random seeds, and dataset switches.
"""

from pathlib import Path
import os

# ==========================================
# Reproducibility
# ==========================================
RANDOM_SEED: int = 42

# ==========================================
# Paths
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MIND_DIR = RAW_DATA_DIR / "mind"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
REPORT_DIR = BASE_DIR / "report"

# Ensure essential output directories exist
for directory in [
    RAW_DATA_DIR,
    MIND_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR,
    REPORT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# Dataset Configuration
# ==========================================
# Options: "mind" | "synthetic"
DATA_SOURCE: str = os.environ.get("DATA_SOURCE", "mind")

# Sampling for laptop performance (MIND dataset can have 50k+ users)
# Setting default to 5,000 users as requested in prompt specification
SAMPLE_USERS: int = 5000
SAMPLE_ITEMS: int = 3000

# Synthetic dataset fallback generator parameters
SYNTHETIC_CONFIG = {
    "num_posts": 2000,
    "num_topics": 10,
    "num_publications": 20,
    "num_readers": 3000,
    "target_interactions": 100000,
}

# MIND small download URLs
MIND_TRAIN_URL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
MIND_DEV_URL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"

# ==========================================
# Evaluation & Splits
# ==========================================
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.10
TOP_K_LIST = [5, 10, 20]
DEFAULT_TOP_K: int = 10

# Plotting settings
PLOT_STYLE = "seaborn-v0_8-whitegrid"
DPI = 150
