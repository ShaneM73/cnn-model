import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to training/
import numpy as np
from PIL import Image, ImageDraw
import train as T
from data import normalize
P = dict(np.load("params.npz"))
def line_img(theta):
    img = Image.new("L", (280, 280), 0); d = ImageDraw.Draw(img)
    r = 110; c = 140; t = np.deg2rad(theta)
    # draw a fixed square frame-limited line; normalization would rescale, so draw directly at 28px instead
    A = np.zeros((28, 28), np.float32)
    big = Image.new("L", (280, 280), 0); d = ImageDraw.Draw(big)
    d.line([(c - r*np.cos(t), c + r*np.sin(t)), (c + r*np.cos(t), c + r*np.sin(t)*-1)], fill=255, width=20)
    a = np.asarray(big, np.float32)/255
    return a.reshape(28,10,28,10).mean((1,3))
angs = np.arange(0, 180, 15)
R = []
for th in angs:
    x = line_img(th)[None, None]
    z1, _ = T.conv_f(x, P["W1"], P["b1"]); a1 = np.maximum(z1, 0)
    R.append(a1[0, :, 6:22, 6:22].mean(axis=(1, 2)))
R = np.array(R).T
np.set_printoptions(linewidth=200)
print(angs)
for ch, r in enumerate(R): print(ch, np.round(r / r.max(), 2), "peak", angs[r.argmax()])
np.save("tuning.npy", R)
