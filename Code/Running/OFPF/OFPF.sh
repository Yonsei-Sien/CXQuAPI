#!/bin/bash
# Every classical path-filter run: the three systems that have one, then the scans across them.
# Nothing here builds a circuit and nothing here needs a solver, so one environment does it all.
# There is no Chain8 driver: the chain is in the study to show that the construction runs at
# d = 8, and its only measurement is a readout, which is a circuit-side run.
#
# Each run appends to Output/<System>/logs/OFPF.log and also prints to the terminal.
set -e
set -o pipefail
cd "$(dirname "$0")"

MASTER=/home/bjbowling123/Project/CX_QuAPI

run () {                       # run <System> <driver> [step flags...]
    local sysname=$1 driver=$2; shift 2
    local dir="$MASTER/Output/$sysname/logs"
    mkdir -p "$dir"
    echo "### $driver $* -> $dir/OFPF.log"
    { echo; echo "### $(date -Is)  $driver $*"; } >> "$dir/OFPF.log"
    python3 "$driver" "$@" 2>&1 | tee -a "$dir/OFPF.log"
}

run TLS        TLS.py        "$@"
run TLS_strong TLS_strong.py "$@"
run Sim2001    Sim2001.py    "$@"
run Common     Common.py     "$@"
