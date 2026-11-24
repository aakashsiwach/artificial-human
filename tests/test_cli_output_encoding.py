"""Regression test: CLI reports must print on consoles that are not UTF-8.

Before the fix, ``artificial-human cognition-eval`` (and ``memory-eval``) raised
``UnicodeEncodeError`` when stdout used a legacy code page such as cp1252,
because reports contain characters like "−" (U+2212) and "–".
"""

from __future__ import annotations

import io
import sys

from artificial_human.cli import main


def _cp1252_stdout() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


def test_cognition_eval_markdown_prints_on_cp1252_stdout(monkeypatch) -> None:
    stream = _cp1252_stdout()
    monkeypatch.setattr(sys, "stdout", stream)
    assert main(["cognition-eval", "--suite", "prediction"]) == 0
    stream.flush()
    text = stream.buffer.getvalue().decode("utf-8")
    assert text.startswith("# Cognitive evaluation report")
    assert "−" in text or "–" in text  # non-cp1252 characters survived


def test_memory_eval_markdown_prints_on_cp1252_stdout(monkeypatch) -> None:
    stream = _cp1252_stdout()
    monkeypatch.setattr(sys, "stdout", stream)
    assert main(["memory-eval", "--backend", "none"]) == 0
    stream.flush()
    assert stream.buffer.getvalue().decode("utf-8").startswith("# Memory retrieval benchmark")
