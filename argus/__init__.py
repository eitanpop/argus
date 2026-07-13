"""Argus — textual-gradient descent over a staged RAG cascade.

A strong LLM plays the role of a gradient-based optimizer: it reads the intermediate
artifacts of a RAG pipeline ("activations") plus an LLM-judged composite-F1 loss, and
emits a "textual gradient" — which knob to move, which way, and why. Apply, re-run,
re-grade, repeat, until the loss plateaus.

The package is backend-agnostic. The bundled showcase is a self-contained toy RAG
pipeline whose knobs mirror a production system 1:1, plus an engineered corpus where the
deliberately-broken default parameters visibly fail and then converge to high F1.
"""

__version__ = "0.1.0"
