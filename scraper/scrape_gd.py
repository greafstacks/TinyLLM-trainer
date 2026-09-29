"""
scrape_gd.py

Scrapes comments from a Geometry Dash level (the current Daily Level by
default) via gd_api.py and writes/merges them into a JSON file in the
same {"levels": {id: {level_id, name, comments: [...]}}} shape as your
earlier dc_2.json/jjj.json exports, so filter_comments.py and
json_to_corpus.py keep working unchanged on the output.

Safe to run daily: comments are merged into any existing file by
comment_id, so re-scraping the same (still-current) daily level twice
doesn't create duplicates.

Usage:
    python scrape_gd.py                          # today's daily level
    python scrape_gd.py --level-id 128            # a specific level instead
    python scrape_gd.py --out scraped/history.json --delay 2
"""

import argparse
import json
import os
import sys

import gd_api


def load_existing(path):
    if not os.path.exists(path):
        return {"levels": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge(data, level_id, name, comments):
    key = str(level_id)
    level = data["levels"].setdefault(key, {"level_id": level_id, "name": name, "comments": []})
    level["name"] = name  # keep it current in case a level was renamed
    seen = {c.get("comment_id") for c in level["comments"] if c.get("comment_id") is not None}
    added = 0
    for c in comments:
        if c["comment_id"] is not None and c["comment_id"] in seen:
            continue
        level["comments"].append(c)
        seen.add(c["comment_id"])
        added += 1
    level["comment_count"] = len(level["comments"])
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level-id", type=int, default=None,
                        help="scrape this level instead of the daily level")
    parser.add_argument("--which", choices=["daily", "weekly", "event"], default="daily",
                        help="ignored if --level-id is given")
    parser.add_argument("--out", default="scraped/daily.json")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between page requests")
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--count", type=int, default=100, help="comments per page (server default is 10)")
    args = parser.parse_args()

    which_code = {"daily": -1, "weekly": -2, "event": -3}[args.which]

    if args.level_id is not None:
        level_id, name = args.level_id, ""
    else:
        print(f"looking up the current {args.which} level...")
        try:
            level_id, name = gd_api.get_level_id_and_name(which_code)
        except gd_api.GDApiError as e:
            print(f"could not look up the {args.which} level: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"level {level_id} ({name!r})")

    def on_page(page, page_count, total):
        print(f"  page {page}: {page_count} comments (running total {total})", flush=True)

    try:
        comments = gd_api.get_all_comments(
            level_id, delay=args.delay, max_pages=args.max_pages, count=args.count, on_page=on_page
        )
    except gd_api.GDApiError as e:
        print(f"stopped early: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"fetched {len(comments)} comments")

    data = load_existing(args.out)
    added = merge(data, level_id, name, comments)
    print(f"{added} new (deduplicated against {args.out} if it already existed)")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
