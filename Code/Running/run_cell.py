"""Run one cell of CXQUAPI.ipynb or OFPF.ipynb without a Jupyter kernel.

The notebooks are the readable form of the drivers: one cell per step, with the settings of that step
written out above the call. A compute node has no kernel, so this runs a cell the way Jupyter would --
the Master cell first, for the import path and the system definitions, then the cell asked for. The
two forms write the same files to the same places, because both only set module globals and call the
measurement's own main().

    python3 run_cell.py CXQUAPI --list          # cell index, system and measurement of every cell
    python3 run_cell.py CXQUAPI 29              # Master, then cell 29
    CXQUAPI_ROOT=/tmp/t python3 run_cell.py OFPF 5    # the same, writing into an empty tree

  [Input]  argv : notebook name (CXQUAPI or OFPF) and a cell index, or --list
           env  : CXQUAPI_ROOT moves Data/ and Output/, exactly as it does for the drivers
  [Output] whatever that cell writes, under Data/<System>/<Property>/
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def cells_of(name):
    with open(os.path.join(HERE, name + '.ipynb'), encoding='utf-8') as fh:
        return json.load(fh)['cells']


# Index, heading and measurements of every code cell, so that a cell can be named without opening the file
def listing(cells):
    heading = ''
    for i, c in enumerate(cells):
        src = ''.join(c['source'])
        if c['cell_type'] == 'markdown':
            if src.startswith('## '):
                heading = src.splitlines()[0][3:]
            continue
        mods = ', '.join(m for line in re.findall(r'from measure import ([^\n]+)', src)
                         for m in [line.replace(' as psd_mod', '')])
        print(f'{i:>3}  {heading:<46}  {mods}')


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('CXQUAPI', 'OFPF'):
        raise SystemExit(__doc__)
    cells = cells_of(sys.argv[1])
    if sys.argv[2] == '--list':
        listing(cells)
        return 0

    # The Master cell is the first code cell: it sets sys.path, empties sys.argv and defines the systems
    master = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code')
    target = int(sys.argv[2])
    if cells[target]['cell_type'] != 'code':
        raise SystemExit(f'cell {target} is a heading, not a step; run --list')

    g = {'__name__': '__main__'}
    for i in (master, target):
        print(f'--- {sys.argv[1]}.ipynb cell {i} ---', flush=True)
        exec(compile(''.join(cells[i]['source']), f'<{sys.argv[1]}.ipynb cell {i}>', 'exec'), g)
    return 0


if __name__ == '__main__':
    sys.exit(main())
