#!/usr/bin/env python3
"""Copy a CmdStan sampling CSV without its saved warmup draws.

Writes the header comments (with save_warmup set to false, so readers that
use it, like sclc_ppc.warmup_rows, count no warmup rows), the column names,
the post-warmup draws and the trailing comments. Rows with the wrong number
of fields (e.g. a partly written last line of a running chain) are dropped.

Used before CmdStan generate_quantities on nsclc-model-marginal.stan fits:
standalone generated quantities first maps each draw back to the
unconstrained scale, which fails for ordered parameters (xi, logit_sens)
whose neighbouring values are equal at the CSV's printed precision. That
happens in the first warmup iterations of a run from random initial values,
not after warmup.

Usage:
  python strip_warmup.py output/fit_1.csv output/fit_1.sampling.csv
"""

import re
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        lines = f.read().split('\n')
    num_warmup, save_warmup = 1000, False
    for l in lines:
        if not l.startswith('#'):
            break
        m = re.match(r'#\s*num_warmup = (\d+)', l)
        if m:
            num_warmup = int(m.group(1))
        m = re.match(r'#\s*save_warmup = (\S+)', l)
        if m:
            save_warmup = m.group(1) in ('1', 'true')
    skip = num_warmup if save_warmup else 0

    out, n_fields, n_draws, seen = [], None, 0, 0
    for l in lines:
        if l.startswith('#'):
            out.append(re.sub(r'save_warmup = \S+', 'save_warmup = false', l))
        elif not l.strip():
            continue
        elif n_fields is None:
            n_fields = l.count(',')
            out.append(l)
        elif l.count(',') == n_fields:
            seen += 1
            if seen > skip:
                out.append(l)
                n_draws += 1
    with open(dst, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'{src}: dropped {min(seen, skip)} warmup rows, kept {n_draws} draws -> {dst}')


if __name__ == '__main__':
    main()
