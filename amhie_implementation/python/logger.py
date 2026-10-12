"""
Safe logging utility for AMHIE.
Guarantees that I/O errors or closed pipes on stdout/stderr never crash the engine.
"""

import sys

def safe_print(*args, **kwargs):
    """Prints safely, swallowing any OSError/IOError (e.g. Errno 5, broken pipes)."""
    try:
        print(*args, **kwargs)
        if kwargs.get("flush", False):
            sys.stdout.flush()
    except (OSError, IOError, BrokenPipeError):
        pass
