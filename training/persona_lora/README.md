# Persona LoRA fine-tuning

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

# CUDA (LLaMA-Factory QLoRA) — BACKEND=cuda. `run` writes the trainer config; the heavy step
# runs in the LLaMA-Factory container (the trainer isn't in the light wheel):
personavoice-train run --persona hr_interviewer --backend cuda \
  --config configs/hr_interviewer.cuda.yaml --dry-run     # writes models/adapters/<p>/<p>_lora.cuda.yaml
docker run --rm --gpus all -v "$PWD":/workspace -w /workspace \
  -v persona-voice_hf-cache:/root/.cache/huggingface hiyouga/llamafactory:latest \
  llamafactory-cli train models/adapters/hr_interviewer/hr_interviewer_lora.cuda.yaml
```

Sharing the `persona-voice_hf-cache` volume reuses the base weights vLLM already downloaded (no
re-download). **On a Blackwell GPU (sm_120)** the stock `hiyouga/llamafactory:latest`
won't run — its torch 2.6.0/cu124 only supports up to sm_90. Build the overlay once and use it as
the image instead:

```
docker build -f training/persona_lora/Dockerfile.blackwell -t personavoice/trainer:blackwell .
# …then `personavoice/trainer:blackwell` in the `docker run` above (Windows Git Bash: prefix
# with MSYS_NO_PATHCONV=1 and use an explicit D:/… path so /workspace isn't path-mangled).
```

An Ada/Ampere card (sm_86 / sm_89) runs the stock image as-is. Adapters land in `<models>/adapters/<persona>/`.

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
personavoice-train eval --persona hr_interviewer --compare --bare-prompt   # the discriminating one
```

Scores persona adherence with deterministic proxies (turn-style fit, spoken-clean rate, question
rate vs `follow_up_probability`, optional keyword coverage) and prints prompt-only-vs-LoRA deltas.
A first-pass signal; pair with human spot-checks for the acceptance bar.

A strong instruct base with the **full** persona prompt is already near-ceiling on these proxies, so
the LoRA shows up as parity. `--bare-prompt` sends only the authored `system_prompt` (dropping the
turn-style + "speak without markdown" directives `render_system_prompt` adds) — there the LoRA wins,
which is the point: it *internalizes* the behavior that otherwise needs explicit prompting. Pass
`--probes <file>` for in-domain prompts and `--keywords a,b,c` to score vocabulary coverage.

## Install

- **Mac:** `pip install -e '.[train]'` (mlx-lm; also in `.[mac]`).
- **CUDA:** use the `hiyouga/llamafactory:latest` Docker image (or the
  `Dockerfile.blackwell` overlay for sm_120 cards) — the trainer needs a CUDA toolchain and is
  heavy, so it stays out of the light `cuda` wheel extra (same policy as vLLM/Chatterbox).
