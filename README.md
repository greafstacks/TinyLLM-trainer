# TinyLLM (trainer)

Trains the chat model for the TinyLLM Android app and produces `model.bin`.

## Pipeline (every file below is regenerated and pushed on each scrape)
1. `daily_chat.py` -- scrapes a level (Daily, or `python daily_chat.py 128`) and ADDS new comments to `scraped/daily.json`.
2. `filter_comments.py` -- `scraped/daily.json` -> `filtered.json` (username + message only).
3. `json_to_corpus.py` -- `filtered.json` -> `corpus.txt` in chat format (`message -> reply`; `--plain` = old flowing text). Uses `chat_data.py`.
4. Pushing `corpus.txt` starts the **Train model** workflow (up to 5h) -> download the `model-bin` artifact -> load it in the app.

`train.py`, `bpe.py`, `model_ref.py` = training, tokenizer, `model.bin` format + self-check.
