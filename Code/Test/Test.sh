#!/bin/bash
# One-system test of CX-QuAPI from pyqake alone. Set PYQAKE to the directory that holds the pyqake/ package.
#   bash Test.sh            # N = 3, g = 2, 3   (seconds)
#   bash Test.sh full       # also N = 6, g = 5, 6: the cover-letter pair
PYQAKE=${PYQAKE:-/home/bjbowling123/Project/PyQAKE/Code/Package}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$PYQAKE${PYTHONPATH:+:$PYTHONPATH}

python3 "$HERE/TLS_strong.py" --N 3 --g 2 3 || exit 1
if [ "$1" == "full" ]; then
    python3 "$HERE/TLS_strong.py" --N 6 --g 5 6 || exit 1
fi
