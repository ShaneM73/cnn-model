import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to training/
import numpy as np, json, time
from numpy.lib.stride_tricks import sliding_window_view as swv

D = np.load("data.npz")
Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
rng = np.random.default_rng(0)
CH = [1, 8, 12, 16]
NC = 5


def he(f, c):
    return (rng.standard_normal((f, c, 3, 3)) * np.sqrt(2 / (c * 9))).astype(np.float32)


P = {"W1": (lambda w: w - w.mean(axis=(2, 3), keepdims=True))(he(8, 1)), "b1": np.zeros(8, np.float32),
     "W2": he(12, 8), "b2": np.zeros(12, np.float32),
     "W3": he(16, 12), "b3": np.zeros(16, np.float32),
     "Wd": (rng.standard_normal((NC, 16)) * np.sqrt(1 / 16)).astype(np.float32), "bd": np.zeros(NC, np.float32)}


def conv_f(x, W, b):
    N, C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = swv(xp, (3, 3), axis=(2, 3))  # N,C,H,W,3,3
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * Wd, C * 9)
    F = W.shape[0]
    out = cols @ W.reshape(F, -1).T + b
    return out.reshape(N, H, Wd, F).transpose(0, 3, 1, 2), cols


def conv_b(dout, cols, x, W):
    N, C, H, Wd = x.shape
    F = W.shape[0]
    d = dout.transpose(0, 2, 3, 1).reshape(-1, F)
    dW = (d.T @ cols).reshape(W.shape)
    db = d.sum(0)
    dcols = (d @ W.reshape(F, -1)).reshape(N, H, Wd, C, 3, 3)
    dxp = np.zeros((N, C, H + 2, Wd + 2), np.float32)
    for i in range(3):
        for j in range(3):
            dxp[:, :, i:i + H, j:j + Wd] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return dxp[:, :, 1:-1, 1:-1], dW, db


def pool_f(x):
    N, C, H, W = x.shape
    r = x.reshape(N, C, H // 2, 2, W // 2, 2)
    m = r.max(axis=(3, 5))
    mask = (r == m[:, :, :, None, :, None])
    return m, mask


def pool_b(d, mask):
    N, C, h, w = d.shape
    g = mask * d[:, :, :, None, :, None]
    g = g / np.maximum(mask.sum(axis=(3, 5), keepdims=True), 1)
    return g.reshape(N, C, h * 2, w * 2)


def forward(x, P, keep=False):
    z1, c1 = conv_f(x, P["W1"], P["b1"]); a1 = np.maximum(z1, 0)
    p1, m1 = pool_f(a1)
    z2, c2 = conv_f(p1, P["W2"], P["b2"]); a2 = np.maximum(z2, 0)
    p2, m2 = pool_f(a2)
    z3, c3 = conv_f(p2, P["W3"], P["b3"]); a3 = np.maximum(z3, 0)
    a3r = a3.reshape(len(a3), a3.shape[1], -1)
    gi = a3r.argmax(2)
    g = np.take_along_axis(a3r, gi[:, :, None], 2)[:, :, 0]
    logits = g @ P["Wd"].T + P["bd"]
    cache = (x, z1, c1, a1, p1, m1, z2, c2, a2, p2, m2, z3, c3, a3, g, gi)
    return logits, cache


def backward(dl, cache, P):
    x, z1, c1, a1, p1, m1, z2, c2, a2, p2, m2, z3, c3, a3, g, gi = cache
    G = {}
    G["Wd"] = dl.T @ g; G["bd"] = dl.sum(0)
    dg = dl @ P["Wd"]
    da3 = np.zeros((len(a3), a3.shape[1], 49), np.float32)
    np.put_along_axis(da3, gi[:, :, None], dg[:, :, None], 2)
    da3 = da3.reshape(a3.shape) * (z3 > 0)
    dp2, G["W3"], G["b3"] = conv_b(da3, c3, p2, P["W3"])
    da2 = pool_b(dp2, m2) * (z2 > 0)
    dp1, G["W2"], G["b2"] = conv_b(da2, c2, p1, P["W2"])
    da1 = pool_b(dp1, m1) * (z1 > 0)
    _, G["W1"], G["b1"] = conv_b(da1, c1, x, P["W1"])
    return G


def shift(xb):
    out = np.zeros_like(xb)
    for i in range(len(xb)):
        dy, dx = rng.integers(-2, 3, 2)
        out[i] = np.roll(np.roll(xb[i], dy, 1), dx, 2)
    return out


def accuracy(X, y):
    pr = []
    for i in range(0, len(X), 500):
        pr.append(forward(X[i:i + 500], P)[0].argmax(1))
    return (np.concatenate(pr) == y).mean()


if __name__ == "__main__":
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    t = 0; lr = 3e-3; B = 64
    EP = 14
    for ep in range(EP):
        t0 = time.time()
        perm = rng.permutation(len(Xtr)); tot = 0
        if ep >= 9: lr = 1e-3
        if ep >= 12: lr = 3e-4
        for s in range(0, len(perm), B):
            idx = perm[s:s + B]
            xb = shift(Xtr[idx]); yb = ytr[idx]
            lg, cache = forward(xb, P)
            lg = lg - lg.max(1, keepdims=True)
            pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
            tot += -np.log(pr[np.arange(len(yb)), yb] + 1e-9).sum()
            dl = pr; dl[np.arange(len(yb)), yb] -= 1; dl /= len(yb)
            G = backward(dl.astype(np.float32), cache, P)
            t += 1
            for k in P:
                M[k] = 0.9 * M[k] + 0.1 * G[k]
                V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                mh = M[k] / (1 - 0.9 ** t); vh = V[k] / (1 - 0.999 ** t)
                P[k] = (P[k] - lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            P["W1"] -= P["W1"].mean(axis=(2, 3), keepdims=True)  # zero-sum filters: respond to edges, not raw ink
        print(f"ep {ep} loss {tot/len(perm):.4f} test {accuracy(Xte, yte):.4f} {time.time()-t0:.0f}s", flush=True)
    np.savez("params.npz", **P)
