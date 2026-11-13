"""Sparse-autoencoder interpretability for AI-designed protein binders.

The scientific code lives in the numbered stage directories at the repository
root (``00_data/`` through ``06_steering/``). Those names are deliberate: they
document the order the pipeline runs in. They are not importable Python
package names, so this package provides the console entry point and dispatches
to each stage script, leaving the stages themselves unchanged.

Run ``sentinel --help`` for the stage list.
"""

__version__ = "0.1.0"
