#!/usr/bin/env python3
"""
daily_chat.py -- standalone (no other files, no pip installs needed).
Scrapes a level's comments from the boomlings API and ADDS them to
scraped/daily.json next to this script. Never removes anything: only new
comments are added (duplicates skipped by comment id).

    python daily_chat.py          # Daily level
    python daily_chat.py 128      # any level ID
"""
import base64
import binascii
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://www.boomlings.com/database"
SECRET = "Wmfd2893gb7"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scraped", "daily.json")


def post(endpoint, data):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(f"{BASE}/{endpoint}", data=body, headers={"User-Agent": ""})
    with urllib.request.urlopen(req, timeout=20) as r:
        text = r.read().decode("utf-8", errors="replace").strip()
    if text in ("-1", "-2", "-10") or text.startswith("temp_"):
        raise RuntimeError(f"{endpoint} returned {text!r}")
    return text


def decode_message(b64):
    for cand in [b64] + [b64[:-n] for n in (1, 2, 3) if len(b64) > n]:
        try:
            s = base64.b64decode(cand + "=" * (-len(cand) % 4)).decode("utf-8", errors="replace")
        except (binascii.Error, ValueError):
            continue
        if s and sum(ch.isprintable() or ch in "\n\t" for ch in s) / len(s) > 0.9:
            return s
    return ""


def kv(segment, sep):
    p = segment.split(sep)
    return {p[i]: p[i + 1] for i in range(0, len(p) - 1, 2)}


def parse_entry(entry):
    cpart, _, upart = entry.partition(":")
    c, u = kv(cpart, "~"), (kv(upart, "~") if upart else {})
    return {
        "username": u.get("1", ""),
        "message": decode_message(c.get("2", "")),
        "comment_id": int(c["6"]) if "6" in c else None,
        "user_id": int(c["3"]) if "3" in c else None,
        "account_id": int(u["16"]) if "16" in u else None,
        "likes": int(c.get("4", 0)),
        "percentage": int(c.get("10", 0)),
        "age": c.get("9", ""),
        "spam": c.get("7") == "1",
    }


def daily_level():
    text = post("downloadGJLevel22.php",
                {"levelID": -1, "secret": SECRET, "gameVersion": 22, "binaryVersion": 42})
    d = kv(text.split("#", 1)[0], ":")
    return int(d["1"]), d.get("2", "")


def all_comments(level_id, delay=1.5, max_pages=300):
    out = []
    for page in range(max_pages):
        text = post("getGJComments21.php", {
            "levelID": level_id, "page": page, "count": 100, "mode": 0, "total": 0,
            "secret": SECRET, "gameVersion": 22, "binaryVersion": 42})
        body = text.split("#", 1)[0]
        if not body:
            break
        for e in body.split("|"):
            try:
                out.append(parse_entry(e))
            except Exception:
                pass
        print(f"  page {page}: total {len(out)}", flush=True)
        time.sleep(delay)
    return out


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    level_id, name = (int(arg), "") if arg.isdigit() else daily_level()
    print(f"level {level_id} {name!r}")
    comments = all_comments(level_id)

    data = {"levels": {}}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            data = json.load(f)
    lv = data["levels"].setdefault(str(level_id), {"level_id": level_id, "name": name, "comments": []})
    if name:
        lv["name"] = name
    seen = {c.get("comment_id") for c in lv["comments"] if c.get("comment_id") is not None}
    added = 0
    for c in comments:
        if c["comment_id"] is not None and c["comment_id"] in seen:
            continue
        lv["comments"].append(c)
        seen.add(c["comment_id"])
        added += 1
    lv["comment_count"] = len(lv["comments"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"{added} new comments added to {OUT}")


if __name__ == "__main__":
    main()
