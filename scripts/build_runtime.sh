#!/bin/sh
# Build runtime/ (third-party dependencies only; the trankit package is loaded from this
# checkout by bin/trankit-run*, so code changes need no reinstall).
#   scripts/build_runtime.sh [python3.10]
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${1:-python3.10}"
"$PYBIN" -c 'import sys; assert sys.version_info[:2]==(3,10)' || { echo "need Python 3.10 (got $PYBIN)" >&2; exit 1; }
"$PYBIN" -m venv "$HERE/runtime"
"$HERE/runtime/bin/python" -m pip install --quiet --upgrade pip
"$HERE/runtime/bin/python" -m pip install --quiet -r "$HERE/requirements-runtime.txt"
"$HERE/runtime/bin/python" -I -c "import sys; sys.path.insert(0, '$HERE'); import trankit; print('ok', trankit.__version__)"
