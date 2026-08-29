"""Collect per-app appearance grades from WebGen-Bench result files.

Reads apps/{app_id}/shots/result{tag}.json files produced by eval_appearance.sh,
extracts the integer grade using the same logic as compute_grade.py's first_grade_int(),
and writes a JSON with per-app grades + summary statistics.
"""

import argparse
import json
import os
import re
import sys


def first_grade_int(text: str) -> int:
    """Return the first integer after 'Grade' (case-insensitive), or 0."""
    match = re.search(r"Grade.*?(\d)", text, flags=re.IGNORECASE | re.DOTALL)
    return int(match.group(1)) if match else 0


def collect_grades(apps_dir: str, app_id_list: list[str], tag: str) -> dict:
    apps = {}
    for app_id in app_id_list:
        # Try both raw ID and zero-padded (e.g. "56" -> "000056")
        app_dir = os.path.join(apps_dir, app_id)
        if not os.path.isdir(app_dir) and app_id.isdigit():
            app_dir = os.path.join(apps_dir, f"{int(app_id):06d}")
        if not os.path.isdir(app_dir):
            print(f"WARNING: app dir not found: {app_dir}", file=sys.stderr)
            apps[app_id] = None
            continue

        # Handle nested single-subdirectory case (same as compute_grade.py's get_app_path)
        entries = os.listdir(app_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(app_dir, entries[0])):
            app_dir = os.path.join(app_dir, entries[0])

        result_path = os.path.join(app_dir, "shots", f"result{tag}.json")
        if not os.path.isfile(result_path):
            print(f"WARNING: result file not found: {result_path}", file=sys.stderr)
            apps[app_id] = None
            continue

        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        grade = first_grade_int(result["model_output"])
        apps[app_id] = grade

    count = len(app_id_list)
    total = sum(v for v in apps.values() if v is not None)
    average = round(total / count, 2) if count > 0 else 0

    return {
        "apps": apps,
        "average": average,
        "total": total,
        "count": count,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect per-app appearance grades.")
    parser.add_argument("--apps-dir", required=True, help="Path to eval apps directory")
    parser.add_argument("--app-id-list", required=True, help="Comma-separated app IDs")
    parser.add_argument("--tag", default="1", help="Result file tag (default: 1)")
    parser.add_argument("--save-path", required=True, help="Output JSON file path")
    args = parser.parse_args()

    app_ids = [x.strip() for x in args.app_id_list.split(",")]
    result = collect_grades(args.apps_dir, app_ids, args.tag)

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Appearance grades: {result['apps']}")
    print(f"Average: {result['average']} (total={result['total']}, count={result['count']})")

    # A crashed appearance run leaves every app ungraded, and the collector then
    # prints "Average: 0.0" — indistinguishable from a portfolio that genuinely
    # scored zero, while the wrapper exits 0 and the caller records a result.
    # Report the failure instead.
    graded = sum(1 for v in result["apps"].values() if v is not None)
    if graded == 0:
        sys.exit(
            f"[error] no app was graded out of {len(app_ids)}: the appearance run "
            f"failed rather than scoring 0. Check the eval log above."
        )
    if graded < len(app_ids):
        missing = [k for k, v in result["apps"].items() if v is None]
        sys.exit(
            f"[error] {graded}/{len(app_ids)} apps graded; missing {missing}. "
            f"The reported average would be computed over a subset."
        )


if __name__ == "__main__":
    main()
