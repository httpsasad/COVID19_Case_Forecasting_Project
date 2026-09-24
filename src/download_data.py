import requests
from pathlib import Path
from .config import DATA_URLS, RAW_DIR

def download_file(url: str, destination: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    destination.write_bytes(r.content)

def main() -> None:
    for name, url in DATA_URLS.items():
        destination = RAW_DIR / f"{name}_global.csv"
        print(f"Downloading {name}...")
        download_file(url, destination)
        print(f"Saved: {destination}")

if __name__ == "__main__":
    main()
