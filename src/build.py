"""Assemble index.html from the template, the app script and the trained-model data.

    python3 src/build.py
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda *p: open(os.path.join(ROOT, *p), encoding="utf-8").read()
t = rd("src", "template.html")
t = t.replace("<!--MODEL-->", rd("training", "model.json").replace("</", "<\\/"))
t = t.replace("<!--FV-->", rd("training", "featvis.json"))
t = t.replace("<!--APPJS-->", rd("src", "app.js"))
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(t)
print("wrote index.html,", len(t), "bytes")
