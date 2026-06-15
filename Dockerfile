# Persona-Voice CUDA server image (RTX 4080 / prod).
#
# Runs the STT (faster-whisper / Parakeet) and TTS (Chatterbox / Orpheus) adapters on the
# GPU. The LLM is served separately by the `vllm` service (see docker-compose.yml), so this
# image deliberately does NOT bundle vLLM — keeping it lighter and the 16 GB VRAM budget
# predictable.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BACKEND=cuda \
    HF_HOME=/models/hf

# Python 3.12 (via deadsnakes) + audio runtime deps: libsndfile (soundfile), ffmpeg (decode).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        libsndfile1 ffmpeg git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package + CUDA extra. Copy metadata/source first so the heavy pip layer is
# cached across edits to config/scripts.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3.12 -m pip install --upgrade pip \
    && python3.12 -m pip install -e '.[cuda]' \
    # GPU TTS: Chatterbox (MIT) is the default — torch ships with it. For Orpheus instead,
    # add `orpheus-speech` (pulls a second in-process vLLM; the Llama-3.2 license also
    # applies, and it's tight on a single 4080 alongside the LLM vLLM service).
    && python3.12 -m pip install chatterbox-tts

COPY config ./config
COPY scripts ./scripts

# Build-time config validation (no weights loaded): a broken backend/persona config fails
# the build rather than the first request.
RUN python3.12 -m personavoice.server --check --no-color

# The live streaming server lands in M3; until then the container validates config and exits.
# docker-compose overrides this with the same command once the vLLM service is healthy.
CMD ["python3.12", "-m", "personavoice.server", "--check", "--no-color"]
