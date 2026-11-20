"""Where this project keeps its numbers.

Data is filed by the system it was measured on and then by the property measured, so that one system
can be read across properties and one property across systems:

    Data/<System>/<Property>/<file>

Nothing measured goes anywhere else. Output/ holds what is made FROM those numbers, and never a
number that no file under Data/ already has:

    Output/Fig/<name>.{pdf,png}      the figures
    Output/CSV/<name>.csv            the numbers each figure draws and each table prints
    Output/<System>/logs/<run>.log   run logs, which are not data

This module is the measurement side only. Analysis/ reads Data/ by writing the path out and keeps its
own CSV writers beside the figure that knows the rows, so the CSV helpers that used to live here were
removed on 2026-11-18 along with `output_path`, which nothing called: a driver's log is redirected by
the shell that starts it.

  [Input]  system : 'TLS', 'TLS_strong', 'Sim2001', ... ; 'Common' for anything spanning systems
           prop   : 'Readout', 'Walsh', 'Grouping', 'LP', 'Cost', 'Bath', 'Shots', 'Validation'
  [Output] an absolute path, with the directory created
"""
import os

# This file is Code/lib/paths.py, so the project root is three levels up. CXQUAPI_ROOT overrides it,
# which is how a full re-run is checked: point it at an empty tree, run Running/, and compare what
# comes out against Data/ here. Nothing else in the tree knows a path, so this one name moves them all.
MASTER_DIR = os.environ.get('CXQUAPI_ROOT') or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Canonical directory name of a system, so that 'Sim2001(beta=0.72)' and 'TLS-strong' file consistently
def system_dir(name):
    n = str(name)
    if n.startswith('Sim2001'):
        return 'Sim2001'
    return n.replace('-', '_').replace('(', '_').replace(')', '').replace('=', '').replace('.', 'p')


def data_path(system, prop, name=''):
    d = os.path.join(MASTER_DIR, 'Data', system_dir(system), prop)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name) if name else d


def fig_path(name=''):
    d = os.path.join(MASTER_DIR, 'Output', 'Fig')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name) if name else d


PAPER_SYSTEMS = ('TLS', 'TLS_strong', 'Sim2001')


# One measurement gathered across systems. Each system keeps its own copy of a study's output under
# Data/<System>/<Property>/, so a measurement that wants all of them concatenates here. A system that
# never ran that measurement contributes nothing.
#   [Input]  name     : file name shared by every system, e.g. 'cost_memory.json'
#            systems  : which systems to gather; the three the manuscript uses by default
#   [Output] the concatenated list of row dictionaries
def read_across(prop, name, systems=PAPER_SYSTEMS):
    import json
    rows = []
    for s in systems:
        p = data_path(s, prop, name)
        if os.path.exists(p):
            rows += json.load(open(p))
    return rows


# A single file, for what is not per system
def read_one(system, prop, name):
    import json
    return json.load(open(data_path(system, prop, name)))
