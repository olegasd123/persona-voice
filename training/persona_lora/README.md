# Persona LoRA fine-tuning (M7)

Train per-persona "brains" beyond prompting. The workflow is one CLI — `personavoice-train` —
backed by `personavoice.training`. Logic is pure and unit-tested; the heavy trainer is shelled
out to (`mlx_lm` on Mac, `llamafactory-cli` on CUDA), so the repo installs and tests without a
GPU or training libs.

```
curate ──▶ (review/edit) ──▶ run ──▶ merge | hot-swap ──▶ eval
```

## 1. Dataset format

One in-character conversation per line, in the OpenAI *messages* shape (the cascade's `Msg`):

```json
{"messages": [
  {"role": "system", "content": "You are a senior HR interviewer ..."},
  {"role": "user", "content": "Hi, thanks for the time."},
  {"role": "assistant", "content": "Of course — glad you're here. ..."}
]}
```

Rules (enforced by `dataset.validate_example`): a single optional leading `system`, then strictly
alternating `user`/`assistant` starting with `user`, ending on `assistant`, no blank content.
This file is consumed directly by mlx-lm's `lora --data <dir>` ("chat" format). For
LLaMA-Factory/Unsloth, convert to ShareGPT (`--fmt sharegpt`).

## 2. Curate seed data

```
personavoice-train curate --persona hr_interviewer --num 20 --exchanges 4
```

Generates dialogues by self-chat (the persona LLM answers; a *user simulator* plays a realistic
partner) and writes `datasets/<persona>/train.jsonl` (+ `valid.jsonl`). **These are a seed —
review and edit before training.** `datasets/hr_interviewer.sample.jsonl` shows the format.

## 3. Train

```
# Mac (mlx-lm light LoRA)
personavoice-train run --persona hr_interviewer --dry-run   # preview the command + config
personavoice-train run --persona hr_interviewer             # launch

# CUDA (LLaMA-Factory QLoRA on the 4080) — BACKEND=cuda
personavoice-train run --persona hr_interviewer --backend cuda
```

Or point at a checked-in config: `--config configs/hr_interviewer.mac.yaml`. Adapters land in
`<models>/adapters/<persona>/`.

### CUDA dataset registration (LLaMA-Factory)

LLaMA-Factory resolves datasets by name via `dataset_info.json` in `dataset_dir`. Register the
ShareGPT file the generated config expects (`<persona>_persona`):

```json
{
  "hr_interviewer_persona": {
    "file_name": "hr_interviewer.sharegpt.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "system": "system" }
  }
}
```

## 4. Serve the adapter (hot-swap) — or merge

**Hot-swap (vLLM, recommended):** launch vLLM with the adapter registered, and a persona whose
`llm.lora` basename matches the served module name is routed to it automatically:

```
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora \
  --lora-modules hr_interviewer=models/adapters/hr_interviewer
```
```yaml
# config/personas/hr_interviewer.yaml
llm:
  lora: hr_interviewer        # was null; basename must match the served module
```

**Merge** (standalone checkpoint for any server):

```
personavoice-train merge --persona hr_interviewer --adapter models/adapters/hr_interviewer
```

## 5. Eval (prompt-only vs LoRA)

```
personavoice-train eval --persona hr_interviewer --compare
```

Scores persona adherence with deterministic proxies (turn-style fit, spoken-clean rate, question
rate vs `follow_up_probability`, optional keyword coverage) and prints prompt-only-vs-LoRA deltas.
A first-pass signal; pair with human spot-checks for the acceptance bar.

## Install

- **Mac:** `pip install -e '.[train]'` (mlx-lm; also in `.[mac]`).
- **CUDA:** install LLaMA-Factory (or Unsloth) in the training image — it needs a CUDA toolchain
  and is heavy, so it stays out of the light `cuda` wheel extra (same policy as vLLM/Chatterbox).
