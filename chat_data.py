"""
chat_data.py -- turns scraped/daily.json (the ONE data file, written by
daily_chat.py) straight into the training text, in memory, every training
run. No filtered.json / corpus.txt to keep in sync.

Format, one exchange per line (U+E000 / U+E001 never appear in comments):
    <U+E000>message<U+E001>reply\\n

  * "@name text" is a real reply to that user's most recent earlier comment.
    These are rare but clean, so they are repeated REPLY_WEIGHT times.
  * any other comment is treated as the answer to the comment before it
    (daily chat is rapid back-and-forth) -- noisier, used once.
"""
import json
import random
import re

USER_MARK = "\ue000"
BOT_MARK = "\ue001"
REPLY_WEIGHT = 3
MAX_CHARS = 200
LOOKBACK = 200

BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
LEADING_MENTION = re.compile(r"^\s*@([A-Za-z0-9_]{1,20})[,:]?\s*")


def _clean(m):
    return " ".join(m.replace(USER_MARK, "").replace(BOT_MARK, "").split())


def _strip_mention(m):
    return LEADING_MENTION.sub("", m, count=1).strip()


def _ordered(level):
    cs = level.get("comments", [])
    if cs and all(c.get("comment_id") is not None for c in cs):
        return sorted(cs, key=lambda c: c["comment_id"])
    return list(reversed(cs))


def build_pairs(data):
    replies, adjacent = [], []
    for level in data.get("levels", {}).values():
        usable = []
        for c in _ordered(level):
            raw = c.get("message", "")
            if raw.strip() and not BASE64_RUN.search(raw):
                usable.append((c.get("username", "").strip().lower(), _clean(raw)))
        for i, (user, msg) in enumerate(usable):
            body = _strip_mention(msg)
            if not body or len(body) > MAX_CHARS:
                continue
            m = LEADING_MENTION.match(msg)
            if m:
                target = m.group(1).lower()
                for j in range(i - 1, max(-1, i - LOOKBACK), -1):
                    if usable[j][0] == target:
                        prompt = _strip_mention(usable[j][1])
                        if prompt and len(prompt) <= MAX_CHARS:
                            replies.append((prompt, body))
                        break
            elif i > 0 and usable[i - 1][0] != user:
                prompt = _strip_mention(usable[i - 1][1])
                if prompt and len(prompt) <= MAX_CHARS:
                    adjacent.append((prompt, body))
    return replies, adjacent


def corpus_from_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    replies, adjacent = build_pairs(data)
    lines = [f"{USER_MARK}{p}{BOT_MARK}{r}" for p, r in replies] * REPLY_WEIGHT
    lines += [f"{USER_MARK}{p}{BOT_MARK}{r}" for p, r in adjacent]
    random.Random(1337).shuffle(lines)
    print(f"data: {len(replies)} real replies (x{REPLY_WEIGHT}) + {len(adjacent)} adjacent pairs "
          f"= {len(lines)} lines from {path}")
    return "\n".join(lines) + "\n"
