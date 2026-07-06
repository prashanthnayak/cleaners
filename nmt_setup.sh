#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nmt"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d "$WORKDIR/MT-Preparation" ]; then
  git clone https://github.com/ymoslem/MT-Preparation.git
fi

# Download dataset
if [ ! -f "$WORKDIR/en-fr.txt.zip" ]; then
  wget https://object.pouta.csc.fi/OPUS-UN/v20090831/moses/en-fr.txt.zip
fi

unzip -o en-fr.txt.zip

# Initial files that are uploaded need to be combined if multiple files into file.src and file.trg
# Adjust the commands below if your archive contains different file names.
if [ ! -f "$WORKDIR/file.src" ] || [ ! -f "$WORKDIR/file.trg" ]; then
  echo "Please combine the downloaded corpus files into file.src and file.trg before continuing." >&2
  exit 1
fi

# Filter
python3 MT-Preparation/filtering/filter.py file.src file.trg src trg

# Train SentencePiece
python3 MT-Preparation/subwording/1-train_unigram.py file.src-filtered.src file.trg-filtered.trg

# Subword
python3 MT-Preparation/subwording/2-subword.py source.model target.model file.src-filtered.src file.trg-filtered.trg

# Split
python3 MT-Preparation/train_dev_split/train_dev_test_split.py 2000 2000 file.src-filtered.src.subword file.trg-filtered.trg.subword
