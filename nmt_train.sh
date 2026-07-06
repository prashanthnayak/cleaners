#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nmt"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Install dependencies
python3 -m pip install --user eole

# Build vocab
if [ -f "$WORKDIR/config.yaml" ]; then
  python3 -m eole build_vocab -config config.yaml -n_sample -1
else
  echo "config.yaml not found in $WORKDIR" >&2
  exit 1
fi

# Train
python3 -m eole train -config config.yaml

# Predict/Translate
if [ -f "$WORKDIR/models/model.fren_step_3000.pt" ]; then
  python3 -m eole predict \
    -model_path models/model.fren_step_3000.pt \
    -src UN.en-fr.fr-filtered.fr.subword.test \
    -output UN.en.translated \
    -gpu_ranks 0 \
    -min_length 1
else
  echo "Trained model not found at $WORKDIR/models/model.fren_step_3000.pt" >&2
  exit 1
fi

# Check translation
head -n 5 UN.en.translated

# Desubword
python3 -m pip install --user --upgrade -q sentencepiece
python3 MT-Preparation/subwording/3-desubword.py target.model UN.en.translated
head -n 5 UN.en.translated.desubword
python3 MT-Preparation/subwording/3-desubword.py target.model UN.en-fr.en-filtered.en.subword.test
head -n 5 UN.en-fr.en-filtered.en.subword.test.desubword

# Evaluate
if [ ! -f "$WORKDIR/compute-bleu.py" ]; then
  wget https://raw.githubusercontent.com/ymoslem/MT-Evaluation/main/BLEU/compute-bleu.py
fi
python3 -m pip install --user sacrebleu
python3 compute-bleu.py UN.en-fr.en-filtered.en.subword.test.desubword UN.en.translated.desubword
