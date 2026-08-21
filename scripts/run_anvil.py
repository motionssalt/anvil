#!/usr/bin/env python3
"""Load the model and launch Anvil outside the notebook."""

from anvil.setup import setup_runtime


if __name__ == "__main__":
    setup_runtime(install=True)
    from anvil.agent import launch
    launch()
