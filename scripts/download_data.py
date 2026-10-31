import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"

DOWNLOADS = [
    ("datasets", "kaggle/us-consumer-finance-complaints", "consumer_complaints.csv"),
    ("competitions", "home-credit-default-risk", "installments_payments.csv"),
    ("competitions", "home-credit-default-risk", "application_test.csv"),
]


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle")
    for kind, ref, file in DOWNLOADS:
        if (EXTERNAL / f"{file}.zip").exists():
            print(f"{file}: вже є")
            continue
        flag = "-c" if kind == "competitions" else ""
        cmd = [str(kaggle), kind, "download", *([flag] if flag else []), ref, "-f", file, "-p", str(EXTERNAL)]
        print(" ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not (EXTERNAL / f"{file}.zip").exists():
            sys.exit(f"Не вдалося завантажити {file}:\n{result.stdout}\n{result.stderr}")
        print(f"{file}: OK")


if __name__ == "__main__":
    main()
