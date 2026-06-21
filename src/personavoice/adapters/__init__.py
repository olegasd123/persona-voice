"""Backend adapters: STT, LLM, and TTS implementations behind stable interfaces.

`base.py` in each subpackage defines a concrete base class adapters inherit from;
`protocols.py` defines the structural contract every backend implements.
`factory.py` maps adapter names from backend YAML to classes.
"""
