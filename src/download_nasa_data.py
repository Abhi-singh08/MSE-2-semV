"""Download and extract the official NASA C-MAPSS archive; keep FD001 only."""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile
import io
import shutil

from .config import DATA_RAW

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
REQUIRED = {"train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"}

def find_and_extract(zip_bytes: bytes, destination: Path, depth: int = 0):
    """Recursively inspect nested ZIPs and extract only the three FD001 files."""
    if depth > 3:
        return set()
    found = set()
    try:
        with ZipFile(io.BytesIO(zip_bytes)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if name in REQUIRED:
                    # Extract only known basenames; no archive paths are trusted.
                    target = destination / name
                    target.write_bytes(archive.read(info))
                    found.add(name)
                elif name.lower().endswith(".zip"):
                    nested = archive.read(info)
                    found |= find_and_extract(nested, destination, depth + 1)
    except BadZipFile:
        return set()
    return found

def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in DATA_RAW.glob("*.txt")}
    if REQUIRED.issubset(existing):
        print("NASA C-MAPSS FD001 files already exist. Nothing to download.")
        return

    print("Downloading official NASA C-MAPSS archive (~12 MB)...")
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response:
        archive_bytes = response.read()

    found = find_and_extract(archive_bytes, DATA_RAW)
    missing = REQUIRED - found
    if missing:
        raise RuntimeError(
            f"Could not find {sorted(missing)} in the downloaded archive. "
            "Download the archive manually from the NASA repository and extract "
            f"the three FD001 files into: {DATA_RAW}"
        )
    print("NASA C-MAPSS FD001 downloaded successfully:")
    for name in sorted(REQUIRED):
        print(f"  - {DATA_RAW / name} ({(DATA_RAW / name).stat().st_size:,} bytes)")
    print("Dataset source: NASA Prognostics Data Repository, Turbofan Engine Degradation Simulation.")

if __name__ == "__main__":
    main()
