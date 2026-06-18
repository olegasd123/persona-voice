#!/usr/bin/env bash
# In-container F5-TTS voice fine-tune (M9), for the Blackwell trainer image.
#
# Reconciles the repo's target-speaker dataset (training/voice/datasets/<DATASET>/, the
# audio_file|text manifest) with f5-tts 1.1.20's actual finetune interface, which differs from
# the older `f5-tts_prepare_csv_wavs` script the plan was written against:
#   * prepare is now a module: `python -m f5_tts.train.datasets.prepare_csv_wavs <CSV> <OUT_DIR>`
#     and wants a CSV with a header line `audio_file|text` and ABSOLUTE wav paths;
#   * the prepared dataset must live at <f5_data>/<DATASET>_<TOKENIZER>/ (get_tokenizer reads
#     its vocab.txt by that name), and finetune writes checkpoints to <f5_ckpts>/<DATASET>/.
# We build the CSV, prepare into F5's data root, fine-tune, then copy the checkpoint back out to
# the mounted repo (the F5 roots live inside the ephemeral container fs).
#
#   docker run --rm --gpus all -v "$PWD":/workspace -w /workspace \
#     -v persona-voice_hf-cache:/models/hf \
#     personavoice/voice-trainer:blackwell bash training/voice/run_finetune.sh
set -euo pipefail

DATASET="${DATASET:-ljspeech}"
TOKENIZER="${TOKENIZER:-pinyin}"
EXP_NAME="${EXP_NAME:-F5TTS_v1_Base}"
DS_DIR="${DS_DIR:-training/voice/datasets/$DATASET}"
OUT_CKPT="${OUT_CKPT:-models/finetuned/$DATASET}"

# Hyperparameters (a short single-speaker fine-tune; frame batching is F5's native mode — the
# repo plan's bare `--batch_size 4` would mean 4 *frames*, so set the type + a real frame count).
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_TYPE="${BATCH_TYPE:-frame}"
BATCH_SIZE="${BATCH_SIZE:-4000}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
EPOCHS="${EPOCHS:-50}"
WARMUP="${WARMUP:-100}"
SAVE_EVERY="${SAVE_EVERY:-300}"
LAST_EVERY="${LAST_EVERY:-150}"

F5_DATA="$(python3.12 -c 'from importlib.resources import files; import os; print(os.path.realpath(str(files("f5_tts").joinpath("../../data"))))')"
F5_CKPTS="$(python3.12 -c 'from importlib.resources import files; import os; print(os.path.realpath(str(files("f5_tts").joinpath("../../ckpts"))))')"
PREPARED="$F5_DATA/${DATASET}_${TOKENIZER}"
CSV="/tmp/${DATASET}_f5.csv"

echo "== F5 fine-tune: dataset=$DATASET tokenizer=$TOKENIZER exp=$EXP_NAME =="
echo "   f5 data root : $F5_DATA"
echo "   f5 ckpt root : $F5_CKPTS"

# 1. Build F5's CSV (header + absolute wav paths) from the repo's audio_file|text manifest.
python3.12 - "$DS_DIR" "$CSV" <<'PY'
import sys, os
ds_dir, csv_path = sys.argv[1], sys.argv[2]
root = os.path.abspath(ds_dir)
rows = []
with open(os.path.join(ds_dir, "metadata.csv"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "|" not in line:
            continue
        rel, text = line.split("|", 1)
        rows.append((os.path.join(root, rel).replace("\\", "/"), text))
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("audio_file|text\n")
    for ap, text in rows:
        f.write(f"{ap}|{text}\n")
print(f"wrote {len(rows)} rows -> {csv_path}")
PY

# 1b. F5 finetune mode asserts the base model's pinyin vocab at <f5_data>/Emilia_ZH_EN_pinyin/
#     vocab.txt (not bundled with the pip package); fetch it from the base model repo.
python3.12 - "$F5_DATA" <<'PY'
import sys, os, shutil
from huggingface_hub import hf_hub_download
f5_data = sys.argv[1]
v = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/vocab.txt")
dst = os.path.join(f5_data, "Emilia_ZH_EN_pinyin")
os.makedirs(dst, exist_ok=True)
shutil.copy(v, os.path.join(dst, "vocab.txt"))
print("pretrained vocab ->", dst)
PY

# 2. Prepare the tokenized dataset into F5's data root (finetune mode → reuses pretrained vocab).
echo "== preparing dataset -> $PREPARED =="
python3.12 -m f5_tts.train.datasets.prepare_csv_wavs "$CSV" "$PREPARED"

# 3. Fine-tune from the base checkpoint (downloads F5TTS_v1_Base into HF_HOME on first run).
echo "== fine-tuning =="
f5-tts_finetune-cli \
  --exp_name "$EXP_NAME" \
  --dataset_name "$DATASET" \
  --tokenizer "$TOKENIZER" \
  --finetune \
  --learning_rate "$LEARNING_RATE" \
  --batch_size_type "$BATCH_TYPE" \
  --batch_size_per_gpu "$BATCH_SIZE" \
  --max_samples "$MAX_SAMPLES" \
  --grad_accumulation_steps 1 \
  --epochs "$EPOCHS" \
  --num_warmup_updates "$WARMUP" \
  --save_per_updates "$SAVE_EVERY" \
  --last_per_updates "$LAST_EVERY" \
  --keep_last_n_checkpoints 1

# 4. Persist the checkpoint(s) back to the mounted repo.
echo "== copying checkpoints -> $OUT_CKPT =="
mkdir -p "$OUT_CKPT"
cp -v "$F5_CKPTS/$DATASET"/*.pt "$OUT_CKPT"/ 2>/dev/null || true
cp -v "$F5_CKPTS/$DATASET"/*.safetensors "$OUT_CKPT"/ 2>/dev/null || true
cp -v "$PREPARED/vocab.txt" "$OUT_CKPT"/ 2>/dev/null || true
ls -la "$OUT_CKPT"
echo "== done =="
