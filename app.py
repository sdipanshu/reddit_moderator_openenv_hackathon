"""
Root-level app.py required by HuggingFace Spaces.

Adds the parent directory to sys.path so that the reddit_mod_env
package is importable, then re-exports the FastAPI app.
"""
import sys
import os

# Ensure the parent of this file (repo root's parent) is on sys.path
# so that `reddit_mod_env` is importable as a package.
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from reddit_mod_env.server.app import app  # noqa: F401

