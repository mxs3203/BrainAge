"""Split the pooled public cache into one CONTIGUOUS file per channel."""
import os
import numpy as np
from build_public_pool import load_pool

HERE = os.path.dirname(os.path.abspath(__file__))


def channel_path(c):
    return os.path.join(HERE, f"public_pool_ch{c:02d}.npy")


def main():
    X, meta = load_pool()
    n = X.shape[0]
    for c in range(X.shape[2]):
        p = channel_path(c)
        if os.path.exists(p):
            print(f"  ch{c:02d} exists"); continue
        out = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n, X.shape[1]))
        step = 20000                        # stream so we never hold teh whole pool
        for i in range(0, n, step):
            out[i:i + step] = X[i:i + step, :, c]
        out.flush(); del out
        print(f"  ch{c:02d} -> {p} ({n * X.shape[1] * 4 / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
