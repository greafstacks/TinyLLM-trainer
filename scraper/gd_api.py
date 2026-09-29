"""
gd_api.py

Talks to Geometry Dash's public "Boomlings" API -- the same backend the
game client itself uses to fetch level comments. It's unauthenticated,
undocumented-by-RobTop-but-extensively-reverse-engineered, and is what
countless community tools (GDBrowser and others) already run on. This
touches nothing private: level comments are visible to anyone who opens
the level in-game.

Endpoint/field reference used to write this: https://boomlings.dev
(specifically the getGJComments21, downloadGJLevel22 and Comment Object
pages). The parsing here was checked against the real example responses
published on those pages before ever being pointed at the live API --
see test_gd_api.py.
"""

import base64
import time

import requests

BASE_URL = "https://www.boomlings.com/database"
SECRET = "Wmfd2893gb7"  # the public "Common Secret" every GD client sends -- not a real secret, see boomlings.dev/reference/secrets
HEADERS = {"User-Agent": ""}  # the server rejects normal User-Agent strings; the game client sends none


class GDApiError(RuntimeError):
    """Raised when the server returns an error/ban code instead of data."""


def _post(endpoint, data, timeout=15):
    resp = requests.post(f"{BASE_URL}/{endpoint}", data=data, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.strip()
    if text in ("-1", "-2", "-10") or text.startswith("temp_"):
        raise GDApiError(f"{endpoint} returned an error/ban code: {text!r}")
    return text


def decode_message(b64_text):
    padded = b64_text + "=" * (-len(b64_text) % 4)
    return base64.b64decode(padded).decode("utf-8", errors="replace")


def _parse_kv(segment):
    parts = segment.split("~")
    return {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}


def parse_comment_entry(entry):
    """One '|'-separated entry from getGJComments21: 'comment_kv:user_kv'."""
    comment_part, _, user_part = entry.partition(":")
    c = _parse_kv(comment_part)
    u = _parse_kv(user_part) if user_part else {}
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


def get_level_id_and_name(which=-1):
    """which: -1 daily, -2 weekly, -3 event. Returns (level_id, name)."""
    text = _post("downloadGJLevel22.php", {"levelID": which, "secret": SECRET, "gameVersion": 22, "binaryVersion": 42})
    level_part = text.split("#", 1)[0]
    kv = _parse_kv_colon(level_part)
    return int(kv["1"]), kv.get("2", "")


def _parse_kv_colon(segment):
    parts = segment.split(":")
    return {parts[i]: parts[i + 1] for i in range(0, len(parts) - 1, 2)}


def get_comments_page(level_id, page, count=100, mode=0):
    """Returns a list of parsed comment dicts for one page (empty list at the end)."""
    text = _post("getGJComments21.php", {
        "levelID": level_id, "page": page, "count": count, "mode": mode, "secret": SECRET,
        "gameVersion": 22, "binaryVersion": 42, "total": 0,
    })
    if not text:
        return []
    body = text.split("#", 1)[0]
    if not body:
        return []
    return [parse_comment_entry(e) for e in body.split("|")]


def get_all_comments(level_id, delay=1.5, max_pages=300, count=100, mode=0, on_page=None):
    """Pages through every comment on a level, oldest pagination stopping when a
    page comes back empty. delay is seconds between requests -- boomlings.com is
    a live server real players depend on, so this is deliberately unhurried."""
    all_comments = []
    for page in range(max_pages):
        items = get_comments_page(level_id, page, count=count, mode=mode)
        if not items:
            break
        all_comments.extend(items)
        if on_page:
            on_page(page, len(items), len(all_comments))
        time.sleep(delay)
    return all_comments
