"""Backend adapters: STT, LLM, and TTS implementations behind stable interfaces.

`base.py` in each subpackage defines a concrete base class adapters inherit from;
`protocols.py` defines the structural contract (the snippet in IMPLEMENTATION_PLAN.md).
`factory.py` maps adapter names from backend YAML to classes.
"""
