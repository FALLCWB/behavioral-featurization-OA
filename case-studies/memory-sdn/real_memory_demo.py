"""Real memory case study for Behavioral Featurization.

Unlike the didactic simulation (memory_demo.py), this version operates a REAL
memory space: an anonymous ``mmap`` arena backed by real OS pages. A small
first-fit allocator hands out real blocks; allocating WRITES real bytes, freeing
ZEROES real bytes, and overwriting REWRITES real bytes. Memory mutates randomly
by nature at every step (diffuse overwrites plus balanced small allocations and
frees), and exactly one ACTION fires once per episode:

    ALLOC  - one allocation burst   (a step up in occupied bytes)
    FREE   - one free burst          (a step down in occupied bytes)
    SCAN   - one localized overwrite (occupancy steady, a concentrated change spike)

Crucially, the memory is DUMPED at every step by reading the process's own
``/proc/self/mem`` at the arena's real virtual address. Every behavioral signal
is derived from the raw bytes read back from RAM, not from the allocator's
bookkeeping: the bookkeeping drives the WRITES, the signals come from re-reading
the real memory. From those raw-byte signals the recipe identifies which action
fired, against the natural random background. The successful real-world
application is cited (Lemos et al. 2024, JNSM).
"""
from __future__ import annotations
import os
import sys
import json
import mmap
import ctypes
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pipeline import featurize_signals, run_classification

PAGE = 4096
ARENA_PAGES = 512                 # 2 MiB real arena
ARENA = ARENA_PAGES * PAGE
SPATIAL_BINS = 16                 # ARENA divisible by SPATIAL_BINS
assert (ARENA_PAGES * PAGE) % SPATIAL_BINS == 0, "ARENA must be divisible by SPATIAL_BINS"
BIN_BYTES = ARENA // SPATIAL_BINS
T = 60                            # time steps per episode
ACTIONS = ["ALLOC", "FREE", "SCAN"]
BLK_MIN, BLK_MAX = 1 * PAGE, 4 * PAGE   # real block sizes


class RealArena:
    """A real mmap-backed arena with a first-fit allocator over real bytes.

    All reads in dump() go through /proc/self/mem at the arena's real virtual
    address: a genuine OS memory dump, independent of the allocator state.
    """

    def __init__(self, rng):
        self.rng = rng
        self.mm = mmap.mmap(-1, ARENA)
        self._buf = (ctypes.c_char * ARENA).from_buffer(self.mm)
        self.va = ctypes.addressof(self._buf)        # real virtual address
        self.memfd = os.open("/proc/self/mem", os.O_RDONLY)
        self.holes = [(0, ARENA)]                    # free list: (offset, size)
        self.blocks = {}                             # id -> (offset, size)
        self._next = 0

    def close(self):
        os.close(self.memfd)
        self._buf = None                             # release view before munmap
        self.mm.close()

    def _nonzero_pattern(self, size):
        # bytes in 1..255 so an occupied byte is never zero (free bytes are zero)
        return self.rng.integers(1, 256, size=size, dtype=np.uint8).tobytes()

    def alloc(self, size):
        for i, (off, hsz) in enumerate(self.holes):
            if hsz >= size:
                self.mm[off:off + size] = self._nonzero_pattern(size)
                if hsz == size:
                    self.holes.pop(i)
                else:
                    self.holes[i] = (off + size, hsz - size)
                bid = self._next
                self._next += 1
                self.blocks[bid] = (off, size)
                return bid
        return None                                  # arena full

    def free(self, bid):
        off, size = self.blocks.pop(bid)
        self.mm[off:off + size] = b"\x00" * size     # real free: zero the bytes
        self.holes.append((off, size))
        self._coalesce()

    def overwrite(self, bid):
        off, size = self.blocks[bid]
        self.mm[off:off + size] = self._nonzero_pattern(size)  # same size, new bytes

    def _coalesce(self):
        self.holes.sort()
        merged = []
        for off, size in self.holes:
            if merged and merged[-1][0] + merged[-1][1] == off:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((off, size))
        self.holes = merged

    def live_ids(self):
        return list(self.blocks)

    def dump(self):
        """Read the real arena bytes from /proc/self/mem at the real VA."""
        raw = os.pread(self.memfd, ARENA, self.va)
        return np.frombuffer(raw, dtype=np.uint8)


def _fill_baseline(arena, target_frac=0.40):
    """Allocate real blocks until ~target_frac of the arena is occupied."""
    occupied = 0
    while occupied < target_frac * ARENA:
        size = int(arena.rng.integers(BLK_MIN, BLK_MAX))
        if arena.alloc(size) is None:
            break
        occupied += size


def natural_step(arena):
    """Background random activity on real memory: diffuse overwrites plus balanced
    small allocations and frees, so occupancy random-walks (net ~ 0)."""
    rng = arena.rng
    live = arena.live_ids()
    if live:
        w = int(rng.integers(8, 16))
        for bid in rng.choice(live, size=min(w, len(live)), replace=False):
            arena.overwrite(int(bid))
    for _ in range(int(rng.integers(0, 8))):
        arena.alloc(int(rng.integers(BLK_MIN, BLK_MAX)))
    live = arena.live_ids()
    if live:
        f = int(rng.integers(0, 8))
        for bid in rng.choice(live, size=min(f, len(live)), replace=False):
            arena.free(int(bid))


def _ensure_live(arena, k):
    """Top up allocations so at least k blocks are live, so the FREE/SCAN action
    always has enough material to act on (otherwise the action could silently
    degrade to a no-op under an unlucky background)."""
    while len(arena.blocks) < k:
        if arena.alloc(int(arena.rng.integers(BLK_MIN, BLK_MAX))) is None:
            break


def action_event(action, arena):
    """One discrete real-memory action, modest in size so it competes with noise.
    The action is guaranteed to fire at its intended magnitude (preconditions on the
    live-block count are enforced first)."""
    rng = arena.rng
    if action == "ALLOC":
        k = int(rng.integers(8, 15))
        for _ in range(k):
            arena.alloc(int(rng.integers(BLK_MIN, BLK_MAX)))
    elif action == "FREE":
        k = int(rng.integers(8, 15))
        _ensure_live(arena, k)
        live = arena.live_ids()
        for bid in rng.choice(live, size=min(k, len(live)), replace=False):
            arena.free(int(bid))
    elif action == "SCAN":
        # contiguous run of live blocks overwritten: occupancy steady, change spike
        k = int(rng.integers(14, 23))
        _ensure_live(arena, k)
        live = sorted(arena.live_ids(), key=lambda b: arena.blocks[b][0])
        if len(live) > k:
            i = int(rng.integers(0, len(live) - k))
            run = live[i:i + k]
        else:
            run = live
        for bid in run:
            arena.overwrite(int(bid))


def _signals_from_dumps(dumps):
    """Derive behavioral signals from the sequence of raw-byte memory dumps."""
    counts, deltas, churns, nets, sp_entropy, centroid = [], [], [], [], [], []
    log = []
    prev = None
    for t, cur in enumerate(dumps):
        nz = cur != 0
        counts.append(float(nz.sum()))
        if prev is None:
            deltas.append(0.0); churns.append(0.0); nets.append(0.0)
            sp_entropy.append(0.0); centroid.append(0.0)
        else:
            changed = cur != prev
            added = ((prev == 0) & (cur != 0)).sum()
            removed = ((prev != 0) & (cur == 0)).sum()
            deltas.append(float(changed.sum()))
            churns.append(float(added + removed))
            nets.append(float(int(added) - int(removed)))
            per_bin = changed.reshape(SPATIAL_BINS, BIN_BYTES).sum(axis=1).astype(float)
            tot = per_bin.sum()
            if tot > 0:
                p = per_bin[per_bin > 0] / tot
                sp_entropy.append(float(-(p * np.log(p)).sum()))
                bin_centers = (np.arange(SPATIAL_BINS) + 0.5) * BIN_BYTES
                centroid.append(float((bin_centers * per_bin).sum() / tot))
                # sample a few changed addresses for the figure
                idx = np.flatnonzero(changed)
                for a in idx[:: max(1, idx.size // 6)][:6]:
                    log.append((t, float(a)))
            else:
                sp_entropy.append(0.0)
                centroid.append(centroid[-1] if centroid else 0.0)
        prev = cur
    signals = {
        "count": counts, "delta": deltas, "churn": churns,
        "net": nets, "spatial_entropy": sp_entropy, "centroid": centroid,
    }
    return signals, log


def simulate_episode(action, rng):
    arena = RealArena(rng)
    _fill_baseline(arena)
    t_event = int(rng.integers(max(2, int(0.34 * T)), max(3, int(0.66 * T))))  # scales with window
    dumps = []
    for t in range(T):
        natural_step(arena)
        if t == t_event:
            action_event(action, arena)
        dumps.append(arena.dump())
    arena.close()
    signals, log = _signals_from_dumps(dumps)
    return signals, log, t_event


def build_signals(n_per_action=120, seed=0):
    """Raw per-window behavioral signals (for the contrast / ablation harness)."""
    rng = np.random.default_rng(seed)
    sigs, y = [], []
    for action in ACTIONS:
        for _ in range(n_per_action):
            signals, _, _ = simulate_episode(action, rng)
            sigs.append(signals)
            y.append(action)
    return sigs, np.array(y)


def build_dataset(n_per_action=120, seed=0):
    rng = np.random.default_rng(seed)
    rng_reps = np.random.default_rng(seed + 99991)  # separate stream: reps must not perturb training
    X, y, reps = [], [], {}
    for action in ACTIONS:
        for _ in range(n_per_action):
            signals, _, _ = simulate_episode(action, rng)
            X.append(featurize_signals(signals))
            y.append(action)
        reps[action] = simulate_episode(action, rng_reps)
    return np.array(X), np.array(y), reps


def make_figure(reps, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"ALLOC": "tab:blue", "FREE": "tab:red", "SCAN": "tab:green"}
    for action, (sig, _, te) in reps.items():
        axes[0].plot(np.asarray(sig["count"]) / 1024.0, label=action, color=colors[action], lw=1.8)
        axes[0].axvline(te, color=colors[action], ls=":", alpha=0.5)
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("occupied (KiB, from /proc/self/mem dump)")
    axes[0].set_title("Real memory dump: occupancy over time")
    axes[0].legend()
    for action, (_, log, _) in reps.items():
        if log:
            ts = [t for t, _ in log]
            ad = [a / 1024.0 for _, a in log]
            axes[1].scatter(ts, ad, s=8, color=colors[action], label=action, alpha=0.5)
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("changed address (KiB)")
    axes[1].set_title("Where change occurs (diffuse natural + action signature)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


def main():
    here = os.path.dirname(__file__)
    figdir = os.path.join(here, "..", "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    X, y, reps = build_dataset()
    res = run_classification(X, y)
    print("=== Behavioral Featurization - REAL memory demo (mmap arena, /proc/self/mem dumps) ===")
    print(f"episodes: {len(y)} | classes: {res['labels']} | features per object: {X.shape[1]}")
    print(f"arena: {ARENA_PAGES} pages ({ARENA // 1024} KiB) | dumps via /proc/self/mem")
    print(f"CV accuracy: {res['cv_mean']:.3f} +/- {res['cv_std']:.3f}")
    print(f"confusion (rows=true {res['labels']}, cols=pred): {res['confusion']}")
    print(res["report"])
    figpath = os.path.join(figdir, "F1_real_memory_demo.png")
    make_figure(reps, figpath)
    print("figure saved:", os.path.relpath(figpath, here))
    with open(os.path.join(here, "results_real_memory.json"), "w") as f:
        json.dump({k: v for k, v in res.items() if k != "report"}, f, indent=2)


if __name__ == "__main__":
    main()
