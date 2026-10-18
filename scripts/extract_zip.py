"""
Fast, robust zip extractor for WildlifeReID-10k handling Windows MAX_PATH (>260 chars).
"""
import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm


def to_windows_long_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


def extract_wildlife_zip(zip_path: str = "data/wildlifereid-10k.zip", target_dir: str = "data"):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[Extract] Opening {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.infolist()
        total_files = len(members)
        print(f"[Extract] Found {total_files:,} files in archive.")
        
        extracted = 0
        skipped = 0
        
        with tqdm(total=total_files, desc="Extracting", unit="file") as pbar:
            for member in members:
                # Skip directory entries
                if member.is_dir():
                    pbar.update(1)
                    continue
                    
                filename = member.filename
                dest = os.path.join(target_dir, filename)
                dest_long = to_windows_long_path(dest)
                
                # Check if already extracted with matching size
                if os.path.isfile(dest_long) and os.path.getsize(dest_long) == member.file_size:
                    skipped += 1
                    pbar.update(1)
                    continue
                    
                os.makedirs(os.path.dirname(dest_long), exist_ok=True)
                with open(dest_long, "wb") as f_out:
                    f_out.write(z.read(member))
                    
                extracted += 1
                pbar.update(1)
                
        print(f"\n[Extract] Done! Extracted: {extracted:,}, Skipped (already existed): {skipped:,}")


if __name__ == "__main__":
    extract_wildlife_zip()
