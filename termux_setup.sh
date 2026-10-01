#!/data/data/com.termux/files/usr/bin/bash
# TinyLLM Termux helper: scrape -> build chat corpus -> push to GitHub (which starts training).
#
# First time:   bash termux_setup.sh
# Later:        cd ~/TinyLLM-trainer && bash termux_setup.sh [LEVEL_ID|daily|weekly|event]
#
# No level given = it asks; press Enter for the Daily level.
# GitHub asks for username (greafstacks) and a Personal Access Token as the
# password the first time only (it is remembered after that).

GH_USER="greafstacks"
REPO="TinyLLM-trainer"

set -e
pkg install -y python git >/dev/null 2>&1 || true
pip install -q requests

cd ~
if [ ! -d "$REPO" ]; then
  git clone "https://github.com/$GH_USER/$REPO.git"
fi
cd "$REPO"

# if this script is the one inside the repo, make sure the repo is up to date
git config credential.helper store
git pull --rebase --autostash || true

python gd_chat.py "$@"

git add chat_corpus.txt scraped/daily.json
if git diff --cached --quiet; then
  echo "nothing new to push."
else
  git commit -m "update chat corpus"
  git push
  echo "pushed. Training starts on GitHub Actions (Actions tab -> Train model)."
fi
