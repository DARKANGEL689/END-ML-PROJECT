"""
Script to download public datasets for the Substack Recommendation System.
Downloads the Microsoft News Dataset (MIND-small) with automatic retries,
decompresses TSV files, and falls back to a realistic synthetic generator if offline.
"""

import argparse
import sys
import os
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MIND_DIR,
    RAW_DATA_DIR,
    MIND_TRAIN_URL,
    MIND_DEV_URL,
    RANDOM_SEED
)


class DownloadProgressBar(tqdm):
    """Custom progress bar hook for urllib download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, max_retries: int = 3, timeout: int = 30) -> bool:
    """
    Download a remote file with retries and progress bar.
    
    Args:
        url: Remote URL.
        output_path: Destination local file path.
        max_retries: Maximum download attempts.
        timeout: Socket timeout in seconds.
        
    Returns:
        bool: True if download succeeded, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"[OK] Cache hit: {output_path.name} already exists.")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (Attempt {attempt}/{max_retries})...")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
                urllib.request.urlretrieve(
                    url,
                    filename=str(output_path),
                    reporthook=t.update_to
                )
            print(f"[OK] Successfully downloaded {output_path.name}")
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
            print(f"[WARN] Attempt {attempt} failed: {e}")
            if output_path.exists():
                output_path.unlink()
    return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip archive to the target folder."""
    if not zip_path.exists():
        return False
    try:
        print(f"Extracting {zip_path.name} to {extract_to}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[OK] Extraction complete for {zip_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract {zip_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and verify recommendation datasets.")
    parser.add_argument("--synthetic-only", action="store_true", help="Skip MIND download and generate synthetic dataset only.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist.")
    args = parser.parse_args()

    train_zip = RAW_DATA_DIR / "MINDsmall_train.zip"
    dev_zip = RAW_DATA_DIR / "MINDsmall_dev.zip"
    train_dir = MIND_DIR / "train"
    dev_dir = MIND_DIR / "dev"

    if args.force:
        for p in [train_zip, dev_zip]:
            if p.exists():
                p.unlink()

    if not args.synthetic_only:
        print("=" * 60)
        print("ATTEMPTING TO DOWNLOAD MIND-SMALL DATASET")
        print("=" * 60)
        train_success = download_file(MIND_TRAIN_URL, train_zip)
        dev_success = download_file(MIND_DEV_URL, dev_zip)

        if train_success and dev_success:
            extract_zip(train_zip, train_dir)
            extract_zip(dev_zip, dev_dir)
            print("\n[SUCCESS] MIND-small dataset ready in", MIND_DIR)
            return

        print("\n[NOTICE] Remote download failed or was unavailable.")
        print("[FALLBACK] Initializing synthetic realistic Substack interaction dataset...")
    else:
        print("\n[FLAG] --synthetic-only specified. Generating synthetic Substack dataset...")

    # Fallback to generating synthetic data
    from src.data_loader import generate_synthetic_substack_data
    generate_synthetic_substack_data()
    print("\n[SUCCESS] Datasets initialized.")


if __name__ == "__main__":
    main()
