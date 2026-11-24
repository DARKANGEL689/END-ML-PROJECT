from pathlib import Path
import os
RANDOM_SEED: int = 42
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
MIND_DIR = RAW_DATA_DIR / 'mind'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
MODELS_DIR = RESULTS_DIR / 'models'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
REPORT_DIR = BASE_DIR / 'report'
for directory in [RAW_DATA_DIR, MIND_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
DATA_SOURCE: str = os.environ.get('DATA_SOURCE', 'mind')
SAMPLE_USERS: int = 5000
SAMPLE_ITEMS: int = 3000
SYNTHETIC_CONFIG = {'num_posts': 2000, 'num_topics': 10, 'num_publications': 20, 'num_readers': 3000, 'target_interactions': 100000}
MIND_TRAIN_URL = 'https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip'
MIND_DEV_URL = 'https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip'
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.1
TOP_K_LIST = [5, 10, 20]
DEFAULT_TOP_K: int = 10
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
DPI = 150
