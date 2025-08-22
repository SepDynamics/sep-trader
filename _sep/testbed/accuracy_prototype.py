import csv
from pathlib import Path

def compute_accuracy(path: Path) -> float:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total = len(rows)
        if total == 0:
            return 0.0
        up = sum(1 for r in rows if float(r['Close']) > float(r['Open']))
        return up / total

if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "tests" / "data" / "eurusd_sample.csv"
    accuracy = compute_accuracy(data_path)
    print(f"Accuracy: {accuracy:.2%}")
