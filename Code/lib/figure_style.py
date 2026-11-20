"""The one definition of how a figure in this study looks.

Nature Portfolio geometry and lettering: single column 89 mm, double column 183 mm, sans-serif 5-7 pt,
lines at least 0.25 pt, bold lower-case panel letters, two spines, outward ticks, a recessive grid the
data is drawn over, and vector PDF beside a 600 dpi PNG. Importing this module sets those rcParams.

It lives in lib/ rather than beside the notebooks because both sides of the tree draw: the notebooks
make the published figures, and `measure/CX_readout_rules.py` draws as well as measures. Two copies of
a style drift apart, so there is one, and `Analysis/style.py` only re-exports it.

Also here, because more than one figure reads them the same way: the per-system names and markers
(SYS, SYS_ORDER) so a reader meets the three baths in one order under one name, the two exact dilations
(CONSTR, CONSTR_ORDER) in the order the text introduces them, and `key()`, the legend every panel with
more than one curve uses in place of a floating label a curve can run into.

The four nature_fig{1..4} builders that used to sit here were removed on 2026-11-11: they read
Data/MethodComparison/ and Data/NoiseFourMethods/, neither of which exists in this tree, so nothing
had called them for a long time.

  [Output] MM, SINGLE, DOUBLE, INK, MUTED, GRID, FIG; bare(ax), panel(ax, letter), save(fig, name),
           key(ax, entries); SYS, SYS_ORDER, CONSTR, CONSTR_ORDER
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from lib import paths

# figures are outputs of this project, not inputs; paths.py owns where every output goes
FIG = paths.fig_path()

MM   = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM
INK, MUTED = '#0b0b0b', '#52514e'


# ------------------------------------------------------------ supplementary-figure conventions
# One name, one order and one marker per trial system, shared by every supplementary figure, so that a
# reader meets the three baths in the same sequence and under the same name in each of them.  The order is
# weakest bath first, which is the order in which the figures read: where the functional splits, through to
# where it does not.  `label` is for a legend, `title` for a panel that shows that system alone.
SYS_ORDER = ('TLS', 'TLS-strong', 'Sim2001')
SYS = {
    'TLS':        dict(label='two-level, weak',   marker='o', ls='-',
                       title='weakly coupled two-level system'),
    'TLS-strong': dict(label='two-level, strong', marker='s', ls='--',
                       title='strongly coupled two-level system'),
    'Sim2001':    dict(label='three-state',       marker='^', ls=':',
                       title='three-state model'),
}

# The two exact dilations, in one order everywhere: the construction of this work leads and the prior one
# follows, in the legend as in the text.  Colour is the project's established pair and carries the identity
# across panels; marker and dash repeat it so identity is never colour alone.
CONSTR_ORDER = ('grouped', 'fused')
CONSTR = {
    'grouped': dict(label='grouped (this work)', color='#003876', marker='o', ls='-'),
    'fused':   dict(label='fused (Walsh)',       color='#760038', marker='s', ls='-'),
}


# One legend for every supplementary panel that draws more than one curve, in place of a floating label that
# a curve can run into.  Entries are given in the canonical order by the caller, never cycled.
#   [Input]  entries : list of (colour, linestyle, marker, text), already ordered
#   [Output] the legend is attached to ax
def key(ax, entries, loc='upper left', ncol=1, size=5.6):
    ax.legend(handles=[Line2D([], [], color=c, lw=1.1, ls=ls, marker=m, ms=3.4, mfc='none', mew=0.9,
                              label=t) for c, ls, m, t in entries],
              loc=loc, fontsize=size, frameon=False, handlelength=2.2, handletextpad=0.5,
              labelspacing=0.35, borderpad=0.0, ncol=ncol)
    return ax

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'xtick.minor.size': 1.2, 'ytick.minor.size': 1.2,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'lines.linewidth': 0.9, 'lines.markersize': 3.0, 'lines.markeredgewidth': 0.0,
    'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.borderpad': 0.2,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


GRID = '#dcdcd9'


def bare(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(True, which='major', lw=0.3, color=GRID, zorder=0)                 # recessive: data is drawn over it
    ax.set_axisbelow(True)
    return ax


def panel(ax, letter, dx=-0.22, dy=1.08):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', ha='left')


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'))
    plt.close(fig)
    print(f'{name} written', flush=True)


