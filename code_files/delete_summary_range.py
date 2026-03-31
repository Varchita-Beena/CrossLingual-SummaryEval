import argparse
from pathlib import Path


DEFAULT_BASE_DIR = Path("output/openai/summary")
DEFAULT_START = 1000
DEFAULT_END = 1499


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete a numeric JSON file range from each summary subfolder."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Base directory containing summary folders.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START,
        help="Start file number, inclusive.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=DEFAULT_END,
        help="End file number, inclusive.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, the script only prints matches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    folders = sorted(path for path in args.base_dir.iterdir() if path.is_dir())
    if not folders:
        print(f"No folders found in {args.base_dir}")
        return

    total_matches = 0
    total_deleted = 0

    for folder in folders:
        folder_matches = 0
        folder_deleted = 0

        for index in range(args.start, args.end + 1):
            file_path = folder / f"{index}.json"
            if not file_path.exists():
                continue

            folder_matches += 1
            total_matches += 1
            print(file_path)

            if args.apply:
                file_path.unlink()
                folder_deleted += 1
                total_deleted += 1

        print(
            f"{folder.name}: matched={folder_matches}"
            + (f" deleted={folder_deleted}" if args.apply else "")
        )

    if args.apply:
        print(f"Done. matched={total_matches} deleted={total_deleted}")
    else:
        print(f"Dry run only. matched={total_matches}")


if __name__ == "__main__":
    main()
