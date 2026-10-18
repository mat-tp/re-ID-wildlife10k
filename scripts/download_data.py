"""
WildlifeReID-10k Dataset Downloader & Validator.

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --data-dir data
"""
import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path


def ensure_kaggle_credentials():
    """Ensure Kaggle credentials are available either via environment or token files."""
    token_file = Path.home() / ".kaggle" / "access_token"
    json_file = Path.home() / ".kaggle" / "kaggle.json"
    
    if token_file.is_file():
        token = token_file.read_text().strip()
        if token:
            os.environ["KAGGLE_API_TOKEN"] = token
            print(f"[Auth] Found Kaggle access token at {token_file}")
            return True
            
    if "KAGGLE_API_TOKEN" in os.environ:
        print("[Auth] Found KAGGLE_API_TOKEN in environment")
        return True
        
    if json_file.is_file() or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ):
        print("[Auth] Found standard Kaggle credentials")
        return True
        
    print("[Error] No Kaggle credentials found!")
    print("Please ensure ~/.kaggle/access_token or ~/.kaggle/kaggle.json exists.")
    return False


def verify_dataset(data_dir: Path):
    """Verify that metadata.csv and images directory exist and are well-formed."""
    meta_path = data_dir / "metadata.csv"
    images_dir = data_dir / "images"
    
    print("\n--- Verifying Dataset Layout ---")
    if not meta_path.is_file():
        print(f"[Failed] Missing metadata file at: {meta_path}")
        return False
        
    if not images_dir.is_dir():
        print(f"[Failed] Missing images directory at: {images_dir}")
        return False
        
    # Check row count
    import pandas as pd
    try:
        df = pd.read_csv(meta_path, low_memory=False)
        row_count = len(df)
        print(f"[OK] metadata.csv loaded successfully: {row_count:,} rows")
        
        # Check first image exists
        first_rel = df["path"].iloc[0]
        first_full = data_dir / first_rel
        if first_full.is_file():
            print(f"[OK] Sample image found: {first_full}")
        else:
            print(f"[Warning] Sample image not found at: {first_full}")
            
        dataset_count = df["dataset"].nunique() if "dataset" in df.columns else 0
        species_count = df["species"].nunique() if "species" in df.columns else 0
        id_count = df["identity"].nunique() if "identity" in df.columns else 0
        print(f"[OK] Benchmark metrics: {dataset_count} datasets, {species_count} species, {id_count} identities.")
        return True
    except Exception as e:
        print(f"[Error] Failed to read metadata.csv: {e}")
        return False


def normalize_nested_structure(data_dir: Path):
    """If the zip extracted into a subfolder like data/wildlifereid-10k/, move contents up."""
    meta_path = data_dir / "metadata.csv"
    if meta_path.is_file():
        return
        
    for sub in data_dir.iterdir():
        if sub.is_dir() and (sub / "metadata.csv").is_file():
            print(f"[Layout] Found nested folder {sub.name}, moving contents to {data_dir}...")
            for item in sub.iterdir():
                dest = data_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            shutil.rmtree(sub)
            print("[Layout] Normalization complete.")
            break


def download_dataset(data_dir: Path, force=False):
    """Download WildlifeReID-10k using Kaggle API."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if not force and (data_dir / "metadata.csv").is_file() and (data_dir / "images").is_dir():
        print(f"[Skip] Dataset already exists at {data_dir}. Use --force to re-download.")
        return verify_dataset(data_dir)
        
    if not ensure_kaggle_credentials():
        sys.exit(1)
        
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[Kaggle] Successfully authenticated with Kaggle API.")
    except Exception as e:
        print(f"[Error] Kaggle authentication failed: {e}")
        sys.exit(1)
        
    dataset_name = "wildlifedatasets/wildlifereid-10k"
    print(f"\n[Download] Starting download of '{dataset_name}' (~26.4 GB)...")
    print(f"[Download] Destination directory: {data_dir.resolve()}")
    print("[Download] This may take some time depending on your connection. Please wait...")
    
    try:
        api.dataset_download_files(dataset_name, path=str(data_dir), unzip=True, quiet=False)
        print("\n[Download] Download and unzipping finished successfully!")
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        sys.exit(1)
        
    normalize_nested_structure(data_dir)
    return verify_dataset(data_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and verify WildlifeReID-10k dataset.")
    parser.add_argument("--data-dir", default="data", help="Target directory (default: data)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing dataset")
    args = parser.parse_args()
    
    target_dir = Path(args.data_dir)
    if args.verify_only:
        normalize_nested_structure(target_dir)
        success = verify_dataset(target_dir)
        sys.exit(0 if success else 1)
        
    success = download_dataset(target_dir, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
