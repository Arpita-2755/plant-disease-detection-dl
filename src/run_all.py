"""Run the complete pipeline (EDA, training, evaluation) in the right order.

  python -m src.run_all
"""
import subprocess
import sys
import time

from src.config import ROOT

STEPS = ("eda", "train_baseline", "tune", "train_transfer", "evaluate", "gradcam",
         "autoencoder", "fgsm", "bias_probe")

if __name__ == "__main__":
    for step in STEPS:
        t = time.time()
        print(f"\n##### {step} #####", flush=True)
        rc = subprocess.run([sys.executable, "-u", "-m", f"src.{step}"], cwd=ROOT).returncode
        print(f"##### {step}: exit {rc}, {time.time() - t:.0f}s #####", flush=True)
        if rc:
            sys.exit(rc)
