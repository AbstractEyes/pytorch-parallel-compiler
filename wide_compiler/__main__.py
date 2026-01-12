"""
WideCompiler CLI entry point.

Usage:
    python -m wide_compiler test
    python -m wide_compiler benchmark --n 100
    python -m wide_compiler trace --model resblock
    python -m wide_compiler info
"""

import sys

try:
    from .cli import main
except ImportError:
    from wide_compiler.cli import main

if __name__ == '__main__':
    sys.exit(main())