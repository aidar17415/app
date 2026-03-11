"""
Airplane Dataset Downloader
============================
Жүктеп алынатын датасеттер / Datasets available for download:

1. OpenFlights Aviation Dataset (airports, airlines, routes, planes)
   Source: https://openflights.org/data.html  (CC BY 4.0)

2. CIFAR-10 "airplane" class (image dataset via torchvision)
   Source: https://www.cs.toronto.edu/~kriz/cifar.html

Usage / Пайдалану:
    python download_dataset.py                   # OpenFlights (default)
    python download_dataset.py --dataset cifar10 # CIFAR-10 airplane images
    python download_dataset.py --output ./data   # custom output directory
"""

import argparse
import csv
import os
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# OpenFlights dataset files
# ---------------------------------------------------------------------------
OPENFLIGHTS_FILES = {
    "airports.csv": {
        "url": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat",
        "columns": [
            "airport_id", "name", "city", "country", "iata", "icao",
            "latitude", "longitude", "altitude", "timezone", "dst",
            "tz_database_timezone", "type", "source",
        ],
        "description": "7,000+ airports worldwide",
    },
    "airlines.csv": {
        "url": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat",
        "columns": [
            "airline_id", "name", "alias", "iata", "icao",
            "callsign", "country", "active",
        ],
        "description": "6,000+ airlines",
    },
    "planes.csv": {
        "url": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/planes.dat",
        "columns": ["name", "iata", "icao"],
        "description": "250+ aircraft types (plane models)",
    },
    "routes.csv": {
        "url": "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat",
        "columns": [
            "airline", "airline_id", "source_airport", "source_airport_id",
            "destination_airport", "destination_airport_id",
            "codeshare", "stops", "equipment",
        ],
        "description": "67,000+ flight routes",
    },
}


def download_openflights(output_dir: Path) -> None:
    """Download the OpenFlights aviation dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading OpenFlights Aviation Dataset...\n")

    for filename, meta in OPENFLIGHTS_FILES.items():
        dest = output_dir / filename
        print(f"  [{meta['description']}]  ->  {dest}")
        urllib.request.urlretrieve(meta["url"], dest)

        # Add header row to the CSV
        with open(dest, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))

        with open(dest, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(meta["columns"])
            writer.writerows(rows)

    print("\nOpenFlights dataset saved to:", output_dir.resolve())
    print("\nFiles:")
    for filename in OPENFLIGHTS_FILES:
        path = output_dir / filename
        print(f"  {path}  ({path.stat().st_size:,} bytes)")


def download_cifar10_airplanes(output_dir: Path) -> None:
    """Download CIFAR-10 and extract the 'airplane' class (label 0)."""
    try:
        import pickle
        import tarfile
    except ImportError as exc:
        raise SystemExit(f"Missing standard library module: {exc}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    archive = output_dir / "cifar-10-python.tar.gz"

    if not archive.exists():
        print(f"Downloading CIFAR-10 from {cifar_url} ...")
        urllib.request.urlretrieve(cifar_url, archive)
        print("  Download complete.")
    else:
        print(f"  Archive already exists: {archive}")

    print("Extracting CIFAR-10 ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(output_dir)

    # Collect airplane images (label 0) from all batches
    cifar_dir = output_dir / "cifar-10-batches-py"
    airplane_images = []
    batch_files = sorted(cifar_dir.glob("data_batch*"))

    for batch_file in batch_files:
        with open(batch_file, "rb") as fh:
            batch = pickle.load(fh, encoding="bytes")
        labels = batch[b"labels"]
        data = batch[b"data"]
        for img, label in zip(data, labels):
            if label == 0:  # 0 = airplane in CIFAR-10
                airplane_images.append(img)

    print(f"  Found {len(airplane_images)} airplane images across {len(batch_files)} batches.")

    # Save as a simple CSV (flattened pixel values)
    airplane_csv = output_dir / "cifar10_airplanes.csv"
    print(f"  Saving to {airplane_csv} ...")
    with open(airplane_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"pixel_{i}" for i in range(3072)])
        writer.writerows(airplane_images)

    print(f"\nCIFAR-10 airplane dataset saved to: {airplane_csv.resolve()}")
    print(f"  Rows : {len(airplane_images)}")
    print("  Cols : 3072  (32x32 image, 3 channels, flattened)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an Airplane dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["openflights", "cifar10"],
        default="openflights",
        help="Which dataset to download (default: openflights)",
    )
    parser.add_argument(
        "--output",
        default="./data",
        help="Directory to save the dataset (default: ./data)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.dataset == "openflights":
        download_openflights(output_dir)
    elif args.dataset == "cifar10":
        download_cifar10_airplanes(output_dir)


if __name__ == "__main__":
    main()
