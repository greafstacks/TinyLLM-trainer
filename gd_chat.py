#!/usr/bin/env python3
"""
gd_chat.py  --  the all-in-one Termux script.

Scrapes a Geometry Dash level's comments and turns EVERYTHING scraped so far
into a CHAT-format corpus (chat_corpus.txt), so the model learns to REPLY to
a message instead of just continuing it.

Usage:
    python gd_chat.py                  # asks for a level ID; just press Enter
                                       #   (or leave it empty) for the Daily level
    python gd_chat.py 128              # scrape level 128
    python gd_chat.py daily|weekly|event
    python gd_chat.py --no-scrape      # skip scraping, just rebuild the corpus
                                       #   from what's already in scraped/*.json

Flags:
    --out FILE          corpus output (default chat_corpus.txt)
    --json FILE         scrape store (default scraped/daily.json; every level
                        you scrape is kept in it, keyed by level id)
    --replies-only      only use real "@name ..." replies (best quality, ~1k pairs)
    --reply-weight N    repeat real replies N times in the corpus (default 3)
    --delay SECONDS     delay between page requests (default 1.5)

Needs `pip install requests` once on Termux (the scraper uses it).

Corpus format, one exchange per line (U+E000 / U+E001 are private-use
characters that can never appear in a comment):

    <U+E000>message<U+E001>reply\\n

The app sends  U+E000 + what you typed + U+E001  and the model writes the
reply until it emits a newline.

Where the pairs come from:
  * "@name text" is a real reply: the pair is (that user's most recent earlier
    comment, text).
  * Otherwise the comment is treated as an answer to the comment right before
    it (daily-chat conversations are rapid back-and-forth). This is noisier,
    which is why real replies get --reply-weight copies.
"""

import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "scraper"))

USER_MARK = "\ue000"
BOT_MARK = "\ue001"

BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
LEADING_MENTION = re.compile(r"^\s*@([A-Za-z0-9_]{1,20})[,:]?\s*")
MAX_CHARS = 200          # drop very long comments (copypasta, essays)
LOOKBACK = 200           # how far back to search for the user being replied to


def clean(message):
    message = message.replace(USER_MARK, "").replace(BOT_MARK, "")
    return " ".join(message.split())


def strip_mention(message):
    return LEADING_MENTION.sub("", message, count=1).strip()


def ordered_comments(level):
    """Oldest -> newest. Uses comment_id when present; otherwise assumes the
    list is newest-first (how the API and filtered.json store it)."""
    comments = level.get("comments", [])
    if comments and all(c.get("comment_id") is not None for c in comments):
        return sorted(comments, key=lambda c: c["comment_id"])
    return list(reversed(comments))


def build_pairs(data, replies_only=False):
    replies, adjacent = [], []
    for level in data.get("levels", {}).values():
        usable = []  # (username_lower, cleaned_message)
        for c in ordered_comments(level):
            raw = c.get("message", "")
            if not raw.strip() or BASE64_RUN.search(raw):
                continue
            usable.append((c.get("username", "").strip().lower(), clean(raw)))

        for i, (user, msg) in enumerate(usable):
            m = LEADING_MENTION.match(msg)
            body = strip_mention(msg)
            if not body or len(body) > MAX_CHARS:
                continue
            if m:
                target = m.group(1).lower()
                for j in range(i - 1, max(-1, i - LOOKBACK), -1):
                    if usable[j][0] == target:
                        prompt = strip_mention(usable[j][1])
                        if prompt and len(prompt) <= MAX_CHARS:
                            replies.append((prompt, body))
                        break
            elif not replies_only and i > 0 and usable[i - 1][0] != user:
                prompt = strip_mention(usable[i - 1][1])
                if prompt and len(prompt) <= MAX_CHARS:
                    adjacent.append((prompt, body))
    return replies, adjacent


def scrape(level_arg, json_path, delay):
    import gd_api
    import scrape_gd

    if level_arg is None or level_arg == "daily":
        which = -1
    elif level_arg == "weekly":
        which = -2
    elif level_arg == "event":
        which = -3
    else:
        which = None

    if which is not None:
        print("looking up the current level...")
        level_id, name = gd_api.get_level_id_and_name(which)
    else:
        level_id, name = int(level_arg), ""
    print(f"level {level_id} {name!r}")

    comments = gd_api.get_all_comments(
        level_id, delay=delay, count=100,
        on_page=lambda p, n, t: print(f"  page {p}: {n} comments (total {t})", flush=True),
    )
    data = scrape_gd.load_existing(json_path)
    added = scrape_gd.merge(data, level_id, name, comments)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"fetched {len(comments)}, {added} new -> {json_path}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level", nargs="?", help="level ID, or daily/weekly/event (default: daily)")
    ap.add_argument("--out", default="chat_corpus.txt")
    ap.add_argument("--json", default="scraped/daily.json")
    ap.add_argument("--no-scrape", action="store_true")
    ap.add_argument("--replies-only", action="store_true")
    ap.add_argument("--reply-weight", type=int, default=3)
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()

    if not args.no_scrape:
        level_arg = args.level
        if level_arg is None and sys.stdin.isatty():
            level_arg = input("Level ID (Enter = Daily level): ").strip() or None
        if level_arg is not None and level_arg not in ("daily", "weekly", "event") \
                and not level_arg.isdigit():
            sys.exit(f"'{level_arg}' is not a level ID (digits) or daily/weekly/event")
        try:
            data = scrape(level_arg, args.json, args.delay)
        except Exception as e:  # network/API errors shouldn't lose the old data
            print(f"scrape failed ({e}); using whatever is already in {args.json}", file=sys.stderr)
            data = None
    else:
        data = None

    if data is None:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)

    replies, adjacent = build_pairs(data, args.replies_only)
    lines = [f"{USER_MARK}{p}{BOT_MARK}{r}" for p, r in replies] * max(1, args.reply_weight)
    lines += [f"{USER_MARK}{p}{BOT_MARK}{r}" for p, r in adjacent]
    random.Random(1337).shuffle(lines)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(replies)} real replies (x{max(1, args.reply_weight)}) "
          f"+ {len(adjacent)} adjacent pairs = {len(lines)} lines")


if __name__ == "__main__":
    main()
