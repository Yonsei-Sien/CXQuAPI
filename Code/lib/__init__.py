"""Shared modules of the CX-QuAPI study.

Every measurement script and figure notebook of this project imports from here. The quantum
construction itself lives in the package, as `pyqake.ansatz.CXQUAPI`, and the independent classical
path sum it is checked against as `pyqake.ansatz.QUAPI`; this package holds only what is specific to
the study -- the model systems, the grouping program, the on-the-fly filter, the figure style, and
the scans built on them.
"""
