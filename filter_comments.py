#!/usr/bin/env python3
"""
filter_comments.py

Strips a comment-export JSON (like dc_2.json) down to just the
username and message for each comment, dropping comment_id, user_id,
account_id, likes, percentage, timestamp, and date. Standard library
only, so it runs on Termux with no extra installs.

Usage:
    python filter_comments.py [input.json] [output.json] [--keep-empty]

Defaults: input.json = dc_2.json, output.json = filtered.json
By default, comments with both an empty username and empty message
are dropped; pass --keep-empty to keep them.
"""

import json
import sys


def filter_comments(data, keep_empty=False):
    levels = data.get("levels", {})
    total_in = 0
    total_out = 0

    for level in levels.values():
        comments = level.get("comments", [])
        total_in += len(comments)

        trimmed = []
        for c in comments:
            username = c.get("username", "")
            message = c.get("message", "")
            if not keep_empty and not username and not message:
                continue
            trimmed.append({"username": username, "message": message})

        level["comments"] = trimmed
        total_out += len(trimmed)

    return data, total_in, total_out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    input_path = args[0] if len(args) > 0 else "dc_2.json"
    output_path = args[1] if len(args) > 1 else "filtered.json"
    keep_empty = "--keep-empty" in sys.argv

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data, total_in, total_out = filter_comments(data, keep_empty)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    dropped = total_in - total_out
    print(f"Read {total_in} comments, kept {total_out}, dropped {dropped} empty.")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
