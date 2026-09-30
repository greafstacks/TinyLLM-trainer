#!/usr/bin/env python3
"""
json_to_corpus.py

Turns filtered.json (username + message pairs) into a plain-text corpus
with no JSON syntax at all -- ready to drop straight into corpus.txt at
the repo root. Standard library only, runs on Termux.

Usage:
    python json_to_corpus.py [input.json] [output.txt] [flags]

Defaults: input.json = filtered.json, output.txt = corpus.txt

Flags:
    --with-username     prefix each line with "username: "
    --keep-base64       don't drop messages containing a base64-looking blob
    --keep-mentions     don't strip a leading "@username" from messages
    --no-terminal-punct don't add "." to messages with no ending punctuation
"""

import json
import re
import sys

# A run this long of only base64-alphabet characters is almost never real
# words -- it's a troll/hidden-message comment encoding something. These
# pollute a character-level model with non-language "random link" output.
BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

# "@username" (optionally followed by , or : ) at the very start of a message.
# Deliberately @-only, not a bare "name,"/"name:" pattern: chat openers like
# "Well," or "gg:" would be indistinguishable from a mention and wrongly
# stripped, so this only touches the unambiguous @ form.
LEADING_MENTION = re.compile(r"^\s*@[A-Za-z0-9_]{1,20}[,:]?\s*")

TERMINAL_PUNCT = ".!?\u2026\"'\u201d)"


def clean_message(message, strip_mentions, add_terminal_punct):
    if strip_mentions:
        message = LEADING_MENTION.sub("", message, count=1)
    message = " ".join(message.split())  # keep it single-line
    if add_terminal_punct and message and message[-1] not in TERMINAL_PUNCT:
        message += "."
    return message


def extract_lines(data, with_username=False, drop_base64=True, strip_mentions=True, add_terminal_punct=True):
    lines = []
    dropped_base64 = 0
    dropped_empty_after_clean = 0
    for level in data.get("levels", {}).values():
        for c in level.get("comments", []):
            raw_message = c.get("message", "").strip()
            if not raw_message:
                continue
            if drop_base64 and BASE64_RUN.search(raw_message):
                dropped_base64 += 1
                continue
            message = clean_message(raw_message, strip_mentions, add_terminal_punct)
            if not message:
                dropped_empty_after_clean += 1
                continue
            if with_username:
                username = c.get("username", "").strip()
                lines.append(f"{username}: {message}" if username else message)
            else:
                lines.append(message)
    return lines, dropped_base64, dropped_empty_after_clean


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    input_path = args[0] if len(args) > 0 else "filtered.json"
    output_path = args[1] if len(args) > 1 else "corpus.txt"
    with_username = "--with-username" in sys.argv
    drop_base64 = "--keep-base64" not in sys.argv
    strip_mentions = "--keep-mentions" not in sys.argv
    add_terminal_punct = "--no-terminal-punct" not in sys.argv

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines, dropped_base64, dropped_empty = extract_lines(
        data, with_username, drop_base64, strip_mentions, add_terminal_punct
    )

    with open(output_path, "w", encoding="utf-8") as f:
        # Every line already ends in terminal punctuation (see add_terminal_punct
        # above), so joining with a single space reads as flowing text --
        # "comment one. comment two. comment three." -- instead of one
        # comment per line.
        f.write(" ".join(lines))

    print(f"Wrote {len(lines)} lines ({sum(len(l) for l in lines)} chars) to {output_path}")
    if drop_base64:
        print(f"Dropped {dropped_base64} comments containing base64-looking blobs (--keep-base64 to keep them)")
    if dropped_empty:
        print(f"Dropped {dropped_empty} comments that were nothing but a mention (e.g. just \"@someone\")")
    if strip_mentions:
        print("Stripped leading @mentions (--keep-mentions to keep them)")
    if add_terminal_punct:
        print("Added '.' to messages with no ending punctuation, as an end-of-message signal (--no-terminal-punct to disable)")


if __name__ == "__main__":
    main()
