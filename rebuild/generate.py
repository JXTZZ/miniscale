"""Generate a response from a MiniScale checkpoint.

Example:
    uv run python generate.py --checkpoint artifacts/run/sft.pt --prompt "Return only the result: 2+3"
"""

import sys

from miniscale.cli import main


if __name__ == "__main__":
    main(["generate", *sys.argv[1:]])
