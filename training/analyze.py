import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to training/
import numpy as np, json, base64
from PIL import Image, ImageDraw
import train as T

P = dict(np.load("params.npz"))
D = np.load("data.npz")
X, y = D["Xte"], D["yte"]
Xtr, ytr = D["Xtr"][:4000], D["ytr"][:4000]


def acts(Xs):
    lg, c = T.forward(Xs, P)
    x, z1, c1, a1, p1, m1, z2, c2, a2, p2, m2, z3, c3, a3, g, gi = c
    return lg, a1, a2, a3


lg, a1, a2, a3 = acts(X)
print("test acc", (lg.argmax(1) == y).mean(), len(y))
from collections import Counter
print(Counter(zip(y[lg.argmax(1) != y], lg.argmax(1)[lg.argmax(1) != y])))

# RF geometry: input rows covered by unit at grid index i
RF = {1: (lambda i: (i - 1, 3)), 2: (lambda i: (2 * i - 3, 8)), 3: (lambda i: (4 * i - 7, 18))}
MARGIN = {1: 3, 2: 2, 3: 0}
out = {"layers": []}
Xp = np.pad(X[:, 0], ((0, 0), (12, 12), (12, 12)))
for L, A in ((1, a1), (2, a2), (3, a3)):
    N, C, H, W = A.shape
    lay = {"channels": []}
    for ch in range(C):
        a = A[:, ch]
        best = a.reshape(N, -1).max(1)
        arg = a.reshape(N, -1).argmax(1)
        order = np.argsort(-best)
        picks = []
        seen = Counter()
        for n in order:
            seen[y[n]] += 1
            picks.append(n)
            if len(picks) == 8:
                break
        m = MARGIN[L]
        pats = []
        for n in picks:
            r, c = divmod(arg[n], W)
            r0, s = RF[L](r); c0, _ = RF[L](c)
            r0 -= m; c0 -= m; s2 = s + 2 * m
            pat = Xp[n, r0 + 12:r0 + 12 + s2, c0 + 12:c0 + 12 + s2]
            pats.append(base64.b64encode((pat * 255).astype(np.uint8).tobytes()).decode())
        top = order[:200]
        cm = np.bincount(y[top], minlength=5) / 200.0
        lay["channels"].append({"patches": pats, "patchY": [int(y[n]) for n in picks],
                                "top200": [round(float(v), 3) for v in cm],
                                "classMean": [round(float(v), 3) for v in cm]})
    lay["patchSize"] = RF[L](0)[1] + 2 * MARGIN[L]
    lay["margin"] = MARGIN[L]
    # selectivity index
    cms = np.array([[A[y == k][:, ch].mean() for k in range(5)] for ch in range(C)])
    mx = cms.max(1); oth = (cms.sum(1) - mx) / 4
    lay["selectivity"] = round(float(((mx - oth) / (mx + oth + 1e-9)).mean()), 3)
    out["layers"].append(lay)

# linear probes on spatially averaged features of each stage (plus raw pixels averaged = 1 number)
_, b1, b2, b3 = acts(Xtr)


def probe(Ftr, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd
    Wt = np.zeros((Ftr.shape[1], 5)); bt = np.zeros(5)
    for it in range(800):
        z = Ftr @ Wt + bt; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        p[np.arange(len(ytr)), ytr] -= 1; p /= len(ytr)
        Wt -= 0.5 * Ftr.T @ p; bt -= 0.5 * p.sum(0)
    return float(((Fte @ Wt + bt).argmax(1) == y).mean())


out["probe"] = [round(probe(Xtr.reshape(len(Xtr), -1).mean(1, keepdims=True), X.reshape(len(X), -1).mean(1, keepdims=True)), 3), round(probe(Xtr.reshape(len(Xtr), -1), X.reshape(len(X), -1)), 3)]
for B, A in ((b1, a1), (b2, a2), (b3, a3)):
    out["probe"].append(round(probe(B.max(axis=(2, 3)), A.max(axis=(2, 3))), 3))
out["testAcc"] = float((lg.argmax(1) == y).mean()); out["testN"] = int(len(y))
print("probe", out["probe"], "selectivity", [l["selectivity"] for l in out["layers"]])

out["weights"] = {k: np.round(v.astype(float), 5).ravel().tolist() for k, v in P.items()}
json.dump(out, open("model.json", "w"), separators=(",", ":"))

# montages for naming
for li, lay in enumerate(out["layers"]):
    s = lay["patchSize"]; z = max(4, 72 // s)
    C = len(lay["channels"])
    img = Image.new("L", ((s * z + 4) * 8 + 140, (s * z + 6) * C), 255)
    d = ImageDraw.Draw(img)
    for ch, c in enumerate(lay["channels"]):
        d.text((2, ch * (s * z + 6) + 4), f"ch{ch} " + " ".join(f"{v:.1f}" for v in c["classMean"]), fill=0)
        for k, b in enumerate(c["patches"]):
            a = np.frombuffer(base64.b64decode(b), np.uint8).reshape(s, s)
            im = Image.fromarray(255 - a).resize((s * z, s * z), Image.NEAREST)
            img.paste(im, (140 + k * (s * z + 4), ch * (s * z + 6)))
            m = lay["margin"] * z
            if m:
                d.rectangle([140 + k * (s * z + 4) + m, ch * (s * z + 6) + m, 140 + k * (s * z + 4) + s * z - m - 1, ch * (s * z + 6) + s * z - m - 1], outline=128)
    img.save(f"top_L{li+1}.png")

# conv1 kernels
W1 = P["W1"][:, 0]
k = np.hstack([np.pad((w - W1.min()) / (W1.max() - W1.min()), 1, constant_values=1) for w in W1])
Image.fromarray((k * 255).astype(np.uint8)).resize((k.shape[1] * 24, k.shape[0] * 24), Image.NEAREST).save("k1.png")
print(np.round(W1, 2))
