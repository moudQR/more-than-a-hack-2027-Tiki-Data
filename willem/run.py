"""CLI: download → train → submit → validate."""
import argparse

import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Possession model trained on Metrica sample data")
    parser.add_argument("step", choices=("download", "train", "submit", "validate", "all"))
    args = parser.parse_args()
    print(f"[run.py] step = {args.step}")

    if args.step in ("download", "all"):
        print("\n=== Downloading Metrica sample matches ===")
        import download_data
        download_data.download()

    if args.step in ("train", "all"):
        print("\n=== Training classifier on Metrica sample matches ===")
        import train
        train.train()

    submission_path = config.RESULTS_DIR / "solution.csv"

    if args.step in ("submit", "all"):
        print("\n=== Building submission ===")
        from make_submission import build_submission
        submission_path = build_submission()

    if args.step in ("validate", "all"):
        print("\n=== Validating submission ===")
        from validate import validate
        if not submission_path.exists():
            from make_submission import build_submission
            submission_path = build_submission()
        validate(submission_path)


if __name__ == "__main__":
    main()