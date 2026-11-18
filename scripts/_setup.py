"""Imported first by every script: reproducible floating-point environment (repro_env.py) and the repository on sys.path."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import repro_env  # noqa: E402,F401  (must run before numpy is imported)
