"""One entry point per system per side, several measurements behind it.

A Running driver names one system and lists the measurements that belong to one side of the
study -- the circuit construction (CXQUAPI) or the classical path filter (OFPF) -- as ordered
(name, thunk) pairs. run_steps() runs them, or the subset the command line names, and prints a
banner before each one so that a run log says which measurement wrote which file.

The order in the table is the order they must run: the only real dependency in this tree is
that `shots` reads what `cost` wrote, and the table keeps them in that order.

  [Input]  steps : ordered ((name, thunk), ...). A thunk takes no argument, sets the module
                   globals its measurement needs, and lets that measurement write its own file.
           argv  : --list            print the step names and exit
                   --only a,b        run only these, in table order
                   --skip a,b        run everything else
                   anything else is left in sys.argv for the measurements to read
  [Output] exit status, 0 when every step ran. Files go wherever each measurement puts them,
           which is Data/<System>/<Property>/ through paths.py.
"""
import sys
import time


def _pick(argv, flag):
    # --only a,b  or  --only a b ... : take the flag and its value out of argv, leaving the rest
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        raise SystemExit(f'{flag} needs a comma-separated list of step names')
    names = [s for s in argv[i + 1].replace(',', ' ').split() if s]
    del argv[i:i + 2]
    return names


def run_steps(steps, argv=None):
    argv = sys.argv if argv is None else argv
    table = dict(steps)

    if '--list' in argv:
        print(' '.join(name for name, _ in steps))
        return 0

    only, skip = _pick(argv, '--only'), _pick(argv, '--skip')
    for name in (only or []) + (skip or []):
        if name not in table:
            raise SystemExit(f'unknown step {name!r}; this driver has: '
                             + ' '.join(n for n, _ in steps))

    chosen = [(n, f) for n, f in steps
              if (only is None or n in only) and (skip is None or n not in skip)]
    if not chosen:
        print('no step selected', flush=True)
        return 0

    for name, thunk in chosen:
        print(f'\n===== {name} =====', flush=True)
        t0 = time.time()
        thunk()
        print(f'===== {name} done in {time.time() - t0:.1f} s =====', flush=True)
    return 0
