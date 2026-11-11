"""Synthetic hand-drawn shapes, rendered on a 280x280 pad and normalized to 28x28
exactly the way the web page does it (bbox -> fit 20px -> area-average)."""
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to training/
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["Plus", "Cross", "Square", "Triangle", "Circle"]
PAD = 280


def normalize(A):
    """A: (280,280) float 0..1 ink. Returns (28,28). Mirrors normalize() in the page."""
    ys, xs = np.nonzero(A > 0.1)
    out = np.zeros((28, 28), np.float32)
    if len(xs) == 0:
        return out
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0, 1)
    s = 20.0 / side
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    yy, xx = np.nonzero(A > 0.001)
    dx = np.floor((xx + 0.5 - cx) * s + 14).astype(int)
    dy = np.floor((yy + 0.5 - cy) * s + 14).astype(int)
    ok = (dx >= 0) & (dx < 28) & (dy >= 0) & (dy < 28)
    np.add.at(out, (dy[ok], dx[ok]), A[yy[ok], xx[ok]] * s * s)
    return np.clip(out, 0, 1)


def wobble(pts, rng, amp):
    """Subdivide a polyline and add smooth low-frequency jitter (hand-drawn feel)."""
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = 12
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(a + (b - a) * t)
    out.append(pts[-1])
    out = np.array(out)
    k = len(out)
    ph = rng.uniform(0, 2 * np.pi, 4)
    t = np.linspace(0, 1, k)
    off = np.stack([np.sin(2 * np.pi * t * 1.3 + ph[0]) + 0.5 * np.sin(2 * np.pi * t * 3.1 + ph[1]),
                    np.sin(2 * np.pi * t * 1.7 + ph[2]) + 0.5 * np.sin(2 * np.pi * t * 2.9 + ph[3])], 1)
    return out + off * amp


def strokes(cls, rng):
    j = lambda s=0.06: rng.uniform(-s, s)
    if cls == 0:  # plus
        cx, cy = 0.5 + j(0.12), 0.5 + j(0.12)
        return [np.array([[cx + j(), 0 + j()], [cx + j(), 1 + j()]]),
                np.array([[0 + j(), cy + j()], [1 + j(), cy + j()]])]
    if cls == 1:  # cross
        return [np.array([[0 + j(), 0 + j()], [1 + j(), 1 + j()]]),
                np.array([[1 + j(), 0 + j()], [0 + j(), 1 + j()]])]
    if cls == 2:  # square / rectangle
        c = [[0 + j(), 0 + j()], [1 + j(), 0 + j()], [1 + j(), 1 + j()], [0 + j(), 1 + j()]]
        c = np.array(c + [c[0]])
        if rng.random() < 0.5:  # drawn as separate sides, with over/undershoot
            segs = []
            for a, b in zip(c[:-1], c[1:]):
                d = b - a
                segs.append(np.array([a - d * rng.uniform(-0.04, 0.1), b + d * rng.uniform(-0.04, 0.1)]))
            return segs
        return [c]
    if cls == 3:  # triangle
        ax = 0.5 + j(0.18)
        c = np.array([[ax, 0 + j()], [1 + j(), 1 + j()], [0 + j(), 1 + j()]])
        c = np.vstack([c, c[:1]])
        if rng.random() < 0.5:
            segs = []
            for a, b in zip(c[:-1], c[1:]):
                d = b - a
                segs.append(np.array([a - d * rng.uniform(-0.04, 0.1), b + d * rng.uniform(-0.04, 0.1)]))
            return segs
        return [c]
    if cls == 4:  # circle / ellipse
        t0 = rng.uniform(0, 2 * np.pi)
        ext = rng.uniform(-0.25, 0.35)
        t = np.linspace(t0, t0 + 2 * np.pi + ext, 48)
        return [np.stack([0.5 + 0.5 * np.cos(t), 0.5 + 0.5 * np.sin(t)], 1)]


def render(cls, rng):
    img = Image.new("L", (PAD, PAD), 0)
    d = ImageDraw.Draw(img)
    size = rng.uniform(120, 230)
    aspect = rng.uniform(0.75, 1.33) if cls in (2, 3, 4) else rng.uniform(0.85, 1.18)
    sx, sy = size * np.sqrt(aspect), size / np.sqrt(aspect)
    ang = np.deg2rad(rng.uniform(-18, 18))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    width = int(round(rng.uniform(0.055, 0.12) * size))
    for st in strokes(cls, rng):
        p = (st - 0.5) * [sx, sy]
        p = wobble(p, rng, size * rng.uniform(0.004, 0.03))
        p = p @ R.T + PAD / 2
        d.line([tuple(q) for q in p], fill=255, width=max(width, 6), joint="curve")
        r = max(width, 6) / 2
        for q in (p[0], p[-1]):
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=255)
    A = np.asarray(img, np.float32) / 255
    return normalize(A)


def make(n_per, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(5):
        for _ in range(n_per):
            X.append(render(c, rng)); y.append(c)
    X = np.array(X)[:, None]
    y = np.array(y)
    p = rng.permutation(len(y))
    return X[p], y[p]


if __name__ == "__main__":
    Xtr, ytr = make(2200, 1)
    Xte, yte = make(300, 2)
    np.savez_compressed("data.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    # montage for eyeballing
    rows = []
    for c in range(5):
        idx = np.nonzero(ytr == c)[0][:12]
        rows.append(np.hstack([np.pad(Xtr[i, 0], 1) for i in idx]))
    m = (np.vstack(rows) * 255).astype(np.uint8)
    Image.fromarray(255 - m).resize((m.shape[1] * 4, m.shape[0] * 4), Image.NEAREST).save("montage.png")
    print(Xtr.shape, Xte.shape)
