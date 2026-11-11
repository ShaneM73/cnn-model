"""Feature visualisation for the trained CNN:
- activation maximisation per channel (what input excites the centre unit most)
- mean of the 50 strongest real patches per channel
- class visualisations (input that maximises each class score)
- knock-out importance: test accuracy per class with one channel zeroed
Writes featvis.json and montage PNGs for inspection."""
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to training/
import numpy as np, json, base64
from PIL import Image
import train as T

P = dict(np.load("params.npz"))
D = np.load("data.npz")
X, y = D["Xte"], D["yte"]
rng = np.random.default_rng(3)
CH = {1: 8, 2: 12, 3: 16}
CENTER = {1: (14, 14, 28), 2: (7, 7, 14), 3: (3, 3, 7)}


def fwd(x, mask=None):
    z1, c1 = T.conv_f(x, P["W1"], P["b1"]); a1 = np.maximum(z1, 0)
    if mask is not None and mask[0] is not None: a1[:, mask[0]] = 0
    p1, m1 = T.pool_f(a1)
    z2, c2 = T.conv_f(p1, P["W2"], P["b2"]); a2 = np.maximum(z2, 0)
    if mask is not None and mask[1] is not None: a2[:, mask[1]] = 0
    p2, m2 = T.pool_f(a2)
    z3, c3 = T.conv_f(p2, P["W3"], P["b3"]); a3 = np.maximum(z3, 0)
    if mask is not None and mask[2] is not None: a3[:, mask[2]] = 0
    a3r = a3.reshape(len(a3), 16, -1); gi = a3r.argmax(2)
    g = np.take_along_axis(a3r, gi[:, :, None], 2)[:, :, 0]
    lg = g @ P["Wd"].T + P["bd"]
    return dict(x=x, z1=z1, c1=c1, a1=a1, p1=p1, m1=m1, z2=z2, c2=c2, a2=a2, p2=p2, m2=m2, z3=z3, c3=c3, a3=a3, g=g, gi=gi, lg=lg)


def grad_input(x, L, k, cls=None):
    """d(objective)/dx. Objective = pre-ReLU value of centre unit of channel k in layer L, or class logit."""
    f = fwd(x)
    if cls is not None:
        dg = np.zeros_like(f["g"]); dg[0] = P["Wd"][cls]
        da3 = np.zeros((1, 16, 49), np.float32)
        np.put_along_axis(da3, f["gi"][:, :, None], dg[:, :, None], 2)
        dz3 = da3.reshape(f["a3"].shape) * (f["z3"] > 0)
        L = 3
    else:
        r, c, H = CENTER[L]
        dz = np.zeros_like(f[f"z{L}"]); dz[0, k, r, c] = 1
        if L == 1:
            _, _, _ = None, None, None
            dx, _, _ = T.conv_b(dz, f["c1"], x, P["W1"]); return dx, f["z1"][0, k, r, c]
        if L == 2: dz2 = dz
        if L == 3: dz3 = dz
    if L == 3:
        dp2, _, _ = T.conv_b(dz3, f["c3"], f["p2"], P["W3"])
        dz2 = T.pool_b(dp2, f["m2"]) * (f["z2"] > 0)
    dp1, _, _ = T.conv_b(dz2, f["c2"], f["p1"], P["W2"])
    dz1 = T.pool_b(dp1, f["m1"]) * (f["z1"] > 0)
    dx, _, _ = T.conv_b(dz1, f["c1"], x, P["W1"])
    val = f["lg"][0, cls] if cls is not None else f[f"z{L}"][0, k, CENTER[L][0], CENTER[L][1]]
    return dx, val


def blur(x, s=0.6):
    k = np.exp(-np.arange(-2, 3) ** 2 / (2 * s * s)); k /= k.sum()
    y = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 2, x[0, 0][None])[0]
    y = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, y)
    return y[None, None]


def maximise(L=None, k=None, cls=None, steps=320):
    best, bv = None, -1e9
    for trial in range(3):
        x = (0.08 + 0.05 * rng.random((1, 1, 28, 28))).astype(np.float32)
        for it in range(steps):
            dy, dxs = rng.integers(-1, 2, 2)
            xs = np.roll(np.roll(x, dy, 2), dxs, 3)
            g, v = grad_input(xs, L, k, cls)
            g = np.roll(np.roll(g, -dy, 2), -dxs, 3)
            x = x + 0.12 * g / (np.abs(g).max() + 1e-8)
            x = x * 0.992  # ink costs a little: keep only what helps
            if it % 3 == 0: x = blur(x, 0.8 if it < steps * 0.8 else 0.5)
            x = np.clip(x, 0, 1).astype(np.float32)
        _, v = grad_input(x, L, k, cls)
        if v > bv: bv, best = v, x
    return best[0, 0], float(bv)


def b64(a):
    a = np.clip(a / max(a.max(), 1e-6), 0, 1)
    return base64.b64encode((a * 255).astype(np.uint8).tobytes()).decode()


out = {"opt": {}, "mean": {}, "cls": [], "ablate": {}, "base": None}
RF = {1: (lambda i: (i - 1, 3)), 2: (lambda i: (2 * i - 3, 8)), 3: (lambda i: (4 * i - 7, 18))}
CROP = {1: 9, 2: 14, 3: 22}  # crop around RF centre for display

# activation maximisation
for L in (1, 2, 3):
    out["opt"][L] = []
    for k in range(CH[L]):
        img, v = maximise(L, k)
        r0, s = RF[L](CENTER[L][0]); cc = r0 + s / 2; h = CROP[L] / 2
        a, b = int(round(cc - h)), int(round(cc - h)) + CROP[L]
        pad = np.pad(img, 6)
        crop = pad[a + 6:b + 6, a + 6:b + 6]
        out["opt"][L].append(b64(crop))
        print("opt", L, k, round(v, 2), flush=True)
    out["opt"][L] = {"size": CROP[L], "rf": RF[L](0)[1], "img": out["opt"][L]}

# mean of top-50 real patches (RF + small margin)
f = fwd(X)
MARG = {1: 3, 2: 3, 3: 2}
Xp = np.pad(X[:, 0], ((0, 0), (12, 12), (12, 12)))
for L in (1, 2, 3):
    A = f[f"a{L}"]; N, C, H, W = A.shape; m = MARG[L]; lst = []
    for k in range(C):
        a = A[:, k].reshape(N, -1); best = a.max(1); arg = a.argmax(1)
        order = np.argsort(-best)[:50]; acc = 0
        for n in order:
            r, c = divmod(arg[n], W); r0, s = RF[L](r); c0, _ = RF[L](c)
            acc = acc + Xp[n, r0 - m + 12:r0 + s + m + 12, c0 - m + 12:c0 + s + m + 12]
        lst.append(b64(acc / len(order)))
    out["mean"][L] = {"size": RF[L](0)[1] + 2 * m, "margin": m, "img": lst}

# class visualisations
for cls in range(5):
    img, v = maximise(cls=cls, steps=400)
    out["cls"].append(b64(img)); print("cls", cls, round(v, 2), flush=True)

# knock-out importance
def acc_per_class(mask=None):
    pr = fwd(X, mask)["lg"].argmax(1)
    return [float((pr[y == c] == c).mean()) for c in range(5)]
out["base"] = acc_per_class()
for L in (1, 2, 3):
    out["ablate"][L] = []
    for k in range(CH[L]):
        m = [None, None, None]; m[L - 1] = [k]
        out["ablate"][L].append([round(v, 3) for v in acc_per_class(m)])
    print("ablate", L, out["ablate"][L], flush=True)
# whole-layer knock-outs of half the channels are not meaningful; also store "only keep top" later in JS
json.dump(out, open("featvis.json", "w"), separators=(",", ":"))

# montages for eyeballing
def dec(s, n): return np.frombuffer(base64.b64decode(s), np.uint8).reshape(n, n) / 255
rows = []
for L in (1, 2, 3):
    o = out["opt"][L]; mm = out["mean"][L]; S = 66
    tiles = []
    for k in range(CH[L]):
        a = Image.fromarray((255 - dec(o["img"][k], o["size"]) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
        b = Image.fromarray((255 - dec(mm["img"][k], mm["size"]) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
        t = Image.new("L", (S, 2 * S + 4), 200); t.paste(a, (0, 0)); t.paste(b, (0, S + 4)); tiles.append(t)
    row = Image.new("L", (16 * (S + 6), 2 * S + 10), 255)
    for i, t in enumerate(tiles): row.paste(t, (i * (S + 6), 3))
    rows.append(row)
cl = Image.new("L", (16 * 72, 120), 255)
for i in range(5): cl.paste(Image.fromarray((255 - dec(out["cls"][i], 28) * 255).astype(np.uint8)).resize((112, 112), Image.NEAREST), (i * 120, 4))
rows.append(cl)
H = sum(r.height for r in rows); M = Image.new("L", (rows[0].width, H), 255); yy = 0
for r in rows: M.paste(r, (0, yy)); yy += r.height
M.save("featvis.png")
