#!/bin/bash
# Every circuit-side run: the four systems, then what is measured across them.
#
# Two things are not negotiable here. `Common.py` runs last because its `shots` step reads the
# gate counts its `cost` step writes, and the system drivers write nothing it needs. And `psd`
# is run on its own, from the virtualenv that has cvxpy, because the semidefinite condition is
# the only measurement in this tree that needs a solver the default environment does not carry.
#
# Each run appends to Output/<System>/logs/CXQUAPI.log and also prints to the terminal.
set -e
set -o pipefail
cd "$(dirname "$0")"

MASTER=/home/bjbowling123/Project/CX_QuAPI
CVX=$HOME/.venvs/cvx/bin/python

run () {                       # run <System> <python> <driver> [step flags...]
    local sysname=$1 python=$2 driver=$3; shift 3
    local dir="$MASTER/Output/$sysname/logs"
    mkdir -p "$dir"
    echo "### $driver $* -> $dir/CXQUAPI.log"
    { echo; echo "### $(date -Is)  $driver $*"; } >> "$dir/CXQUAPI.log"
    "$python" "$driver" "$@" 2>&1 | tee -a "$dir/CXQUAPI.log"
}

run TLS        python3 TLS.py        --skip psd "$@"
run TLS_strong python3 TLS_strong.py --skip psd "$@"
run Sim2001    python3 Sim2001.py    --skip psd "$@"
run Chain8     python3 Chain8.py                "$@"
run Common     python3 Common.py                "$@"

# the semidefinite comparison, in the environment that has cvxpy
run TLS_strong "$CVX" TLS_strong.py --only psd
run TLS        "$CVX" TLS.py        --only psd
run Sim2001    "$CVX" Sim2001.py    --only psd
