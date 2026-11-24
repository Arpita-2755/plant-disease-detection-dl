"""Cross-check the report against metrics.json and look for leftovers.

  python tools/verify_submission.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "metrics.json").read_text())
PDF = ROOT / "report" / "Arpita_Mishra_12303670_INT422_Project_Report.pdf"
doc = pymupdf.open(PDF)
pages = [p.get_text() for p in doc]
text = re.sub(r"\s+", " ", " ".join(pages))
pct = lambda x, d=1: f"{100 * x:.{d}f}%"  # noqa: E731
bad = []


def need(label, value):
    if str(value) not in text:
        bad.append(f"{label}: {value} not found in the report")


n = 0
E = M["evaluation"]
for m in ("baseline", "transfer"):
    e = E[m]
    items = [("accuracy", pct(e["accuracy"], 2)), ("f1", f"{e['macro_f1']:.3f}"), ("precision", f"{e['macro_precision']:.3f}"),
             ("recall", f"{e['macro_recall']:.3f}"), ("correct", f"{e['correct']}/{e['total']}"),
             ("ci", f"{100 * e['wilson95'][0]:.1f} to {100 * e['wilson95'][1]:.1f}%"),
             ("params", f"{M[m]['params']:,}"), ("latency", f"{e['latency_ms']:.0f} ms")]
    items += [(f"f1 {c}", f"{v['f1-score']:.2f}") for c, v in e["per_class"].items()]
    for k, v in items:
        need(f"{m} {k}", v)
        n += 1
for m, accs in M["fgsm"]["accuracy"].items():
    for ep, a in zip(M["fgsm"]["eps"], accs):
        need(f"fgsm {m} eps {ep}", pct(a))
        n += 1
for r in M["tuning"]["table"]:
    need("tuning val loss", f"{r['val_loss']:.3f}")
    need("tuning val acc", pct(r["val_acc"]))
    n += 2
for k, v in M["fit_experiment"]["results"].items():
    for q in ("train_acc", "val_acc"):
        need(f"fit {k} {q}", pct(v[q]))
        n += 1
for sec, keys in (("autoencoder", ("float_code", "uint8_code", "jpeg_same_budget")), ("denoiser", ("noisy", "median3", "gaussian3", "autoencoder")),
                  ("vae", ("recon",))):
    for k in keys:
        need(f"{sec} {k} psnr", f"{M[sec][k]['psnr']:.2f}")
        need(f"{sec} {k} ssim", f"{M[sec][k]['ssim']:.3f}")
        n += 2
need("bias", pct(M["bias_probe"]["accuracy"]))
need("total images", f"{M['data']['total']:,}")
for c, v in M["data"]["counts"].items():
    need(f"class {c}", c)
n += 2 + len(M["data"]["counts"])

for word in ("lorem", "todo", "tbd", "xxx", "placeholder", "error! ", "reference source not found", "update field"):
    if word in text.lower():
        bad.append(f"leftover text: {word}")
blank = [i + 1 for i, p in enumerate(pages) if len(p.strip()) < 40 and not doc[i].get_images()]
if blank:
    bad.append(f"almost blank pages: {blank}")
cites = {int(x) for x in re.findall(r"\[(\d{1,2})\]", text)}
refs = max(cites)
ref_page = next(i for i, p in enumerate(pages) if "[1]" in p and "[2]" in p and "doi:" in p)
body = re.sub(r"\s+", " ", " ".join(pages[:ref_page]))
uncited = [i for i in range(1, refs + 1) if f"[{i}]" not in body]
if uncited:
    bad.append(f"references never cited in the text: {uncited}")
for f in ("README.md", "VIVA_NOTES.md", "SUBMISSION_CHECKLIST.md", "PLAN.md", "Dockerfile", "requirements.txt", "LICENSE",
          "models/baseline.keras", "models/transfer.keras", "models/autoencoder.keras", "models/denoiser.keras",
          "notebooks/plant_disease_workflow.ipynb", "assets/screenshots/demo.gif"):
    if not (ROOT / f).exists():
        bad.append(f"missing file: {f}")
print(f"{len(pages)} pages, {n} numbers checked, {refs} references")
print("\n".join(bad) if bad else "ALL CHECKS PASSED")
sys.exit(bool(bad))
