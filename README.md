# TinyLLM (trainer)

Trains the model for [TinyLLM's Android app](../app-repo) and produces
`model.bin`. No Android/Gradle here -- pure Python.

## Layout

- `corpus.txt` -- training text, one message per line. Put your own
  here (not included in this zip).
- `bpe.py` -- learns a subword vocabulary from the corpus, so the model
  works roughly word-by-word instead of letter-by-letter.
- `train.py` -- trains the transformer with PyTorch (keeping the best
  checkpoint by validation loss) and exports `model.bin`.
- `model_ref.py` -- a from-scratch NumPy reimplementation of the model
  and the exact `model.bin` layout (documented at the top of the file).
  After training, `train.py` re-reads the exported file with this and
  checks it against PyTorch's own output, the KV-cache path the phone
  uses, and the tokenizer -- if anything disagrees, the run fails
  instead of shipping a model the app would run differently.
- `filter_comments.py` / `json_to_corpus.py` -- turn a raw scrape (see
  below) into `corpus.txt`: strip everything except username/message,
  drop base64-blob comments, strip leading `@mentions`, and make sure
  every message ends with punctuation (a "this message is over" signal
  for the model, instead of leaving that entirely to newlines).
- `scraper/` -- pulls comments from a Geometry Dash level via the public
  Boomlings API (the same backend the game client uses; level comments
  are visible to anyone who opens the level in-game). Defaults to the
  current Daily Level.

## Chat mode (replies instead of autocomplete)

`gd_chat.py` scrapes a level (any level ID, or Daily by default) and builds
`chat_corpus.txt` in `message -> reply` format. `termux_setup.sh` runs it on
your phone and pushes the result, which starts training. See the top of each
file for usage. `train.py` now trains on `chat_corpus.txt` for up to 5 hours.

## Workflows

- **Train model** (`train.yml`) -- runs on any push that touches
  `corpus.txt` or the training scripts, or manually from the Actions
  tab. Trains and uploads `model.bin` as a workflow artifact. To update
  the app, download that artifact and replace
  `app/src/main/assets/model.bin` in the app repo with it.
- **Scrape daily level comments** (`scrape.yml`) -- runs once a day on
  a schedule (or manually), scrapes the current Daily Level's comments
  into `scraped/daily.json`, and commits the result. Safe to run
  repeatedly: comments are merged and deduplicated by ID, so scraping
  the same still-current daily level twice doesn't create duplicates.

## Building a corpus from scraped data

```
python filter_comments.py scraped/daily.json filtered.json
python json_to_corpus.py filtered.json corpus.txt
```

`json_to_corpus.py` flags: `--with-username`, `--keep-base64`,
`--keep-mentions`, `--no-terminal-punct` (see its docstring).

To combine with other text (e.g. Gutenberg novels) and rebalance the
mix, `cat` them together -- repeat the smaller file if you want it to
carry more relative weight:

```
cat gutenberg.txt corpus.txt corpus.txt corpus.txt > combined.txt
mv combined.txt corpus.txt
```

## Tuning

See the flags in `train.py` (`--vocab-size`, `--n-embd`, `--n-head`,
`--n-layer`, `--block-size`, `--max-iters`, `--max-minutes`, etc.) --
pass them on the `python -u train.py` line in `train.yml`.
