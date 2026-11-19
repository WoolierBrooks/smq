"""Generate the synthetic curves used to train SMQ, one file per class count.

    python experiments/generate_synthetic.py --out data/synthetic
    python experiments/generate_synthetic.py --classes 3 10 50 --seed 0

Each file ``moss_d500_lite_<C>.pkl`` holds ``{(prevalence, m): [scores, ...]}``
with 15 prevalences x 15 merging factors x 20 samples of 100 instances, the
configuration of the published experiment. The original files were generated
without a fixed seed, so regenerated files are statistically equivalent but
not bit-identical.
"""

import argparse
import os
import pickle
import time

from smq import synthetic_curves

# class counts generated for the paper (exp_010/note.ipynb)
PAPER_CLASSES = [3, 4, 5, 6, 7, 8, 10, 11, 12, 15, 18, 22, 26, 31, 46, 48, 50]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--classes", type=int, nargs="*", default=PAPER_CLASSES)
    ap.add_argument("--n-prevalences", type=int, default=15)
    ap.add_argument("--n-merging", type=int, default=15)
    ap.add_argument("--n-replicates", type=int, default=20)
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    for c in args.classes:
        t0 = time.perf_counter()
        curves = synthetic_curves(
            c,
            n_prevalences=args.n_prevalences,
            n_merging=args.n_merging,
            n_replicates=args.n_replicates,
            sample_size=args.sample_size,
            random_state=args.seed + c,
        )
        path = os.path.join(args.out, f"moss_d500_lite_{c}.pkl")
        with open(path, "wb") as f:
            pickle.dump(curves, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"C={c:>2}: {len(curves)} pairs -> {path} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
