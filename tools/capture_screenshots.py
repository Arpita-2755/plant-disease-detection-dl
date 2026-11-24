"""Start the app and TensorBoard, drive them in headless Chrome, save real screenshots.

  python tools/capture_screenshots.py            app + TensorBoard
  python tools/capture_screenshots.py app        app only
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "assets" / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
APP, TB = 8511, 6011


def wait(url: str, seconds: int = 120) -> None:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"{url} did not start")


def settle(page, extra: float = 1.2) -> None:
    """Wait until Streamlit has finished its script run."""
    page.wait_for_timeout(700)
    page.wait_for_function(
        "() => document.querySelector('[data-testid=\"stApp\"]')"
        "?.getAttribute('data-test-script-state') !== 'running'", timeout=180000)
    page.wait_for_timeout(int(extra * 1000))


def drive_app(browser) -> None:
    import pandas as pd

    page = browser.new_page(viewport={"width": 1360, "height": 940})
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    frames: list[Path] = []

    def shot(name: str, full=False):
        path = SHOTS / f"{name}.png"
        assert page.get_by_test_id("stException").count() == 0, "app raised an exception"
        page.screenshot(path=str(path), full_page=full)
        frames.append(path)
        print("saved", path.name, flush=True)

    def tab(label: str):
        page.get_by_role("tab", name=label).click()
        settle(page)

    page.goto(f"http://localhost:{APP}")
    page.get_by_text("Choose an image to start").wait_for(timeout=180000)
    settle(page)
    shot("01_home")

    # a real upload: a test image that is not one of the bundled samples
    df = pd.read_csv(ROOT / "data" / "manifest.csv")
    row = df[(df.split == "test") & (df.class_dir == "Tomato___Late_blight")].iloc[0]
    page.locator("input[type=file]").set_input_files(str(ROOT / "data/plantvillage" / row.class_dir / row.filename))
    page.get_by_text("Top-3 probabilities").first.wait_for(timeout=180000)
    settle(page)
    shot("02_upload_prediction")
    page.get_by_text("Use a sample leaf").click()
    settle(page)
    page.get_by_test_id("stSelectbox").first.locator("input").click()
    page.get_by_role("option", name="Potato Early blight").click()
    page.get_by_text("Top-3 probabilities").first.wait_for(timeout=120000)
    settle(page)
    shot("03_sample_prediction")

    tab("Grad-CAM")
    shot("04_gradcam")

    tab("FGSM attack")
    slider = page.get_by_role("slider", name="Epsilon").first
    slider.focus()
    for _ in range(4):
        page.keyboard.press("ArrowLeft")       # 0.0100 -> 0.0000
    settle(page)
    page.mouse.move(800, 700)
    page.mouse.wheel(0, 260)
    page.wait_for_timeout(700)
    shot("05_fgsm_eps0")
    for _ in range(8):
        page.keyboard.press("ArrowRight")      # -> 0.0200
    settle(page, 2)
    shot("06_fgsm_attack")

    page.mouse.wheel(0, -2000)
    page.wait_for_timeout(500)
    tab("Autoencoder")
    shot("07_autoencoder", full=False)
    page.get_by_text("Denoising", exact=True).scroll_into_view_if_needed()
    page.mouse.move(900, 600)
    page.mouse.wheel(0, 900)
    page.wait_for_timeout(900)
    shot("08_denoiser")

    page.mouse.wheel(0, -3000)
    page.get_by_test_id("stSidebar").get_by_text("Baseline CNN", exact=True).click()
    settle(page)
    tab("Prediction")
    shot("09_baseline_model")
    if errors:
        print("page errors:", errors)

    ims = [Image.open(f).convert("RGB").resize((1020, 705), Image.LANCZOS) for f in frames]
    pal = [im.quantize(128, method=Image.Quantize.MEDIANCUT) for im in ims]
    pal[0].save(SHOTS / "demo.gif", save_all=True, append_images=pal[1:], duration=1900, loop=0, optimize=True)
    print("saved demo.gif")


def drive_tensorboard(browser) -> None:
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(f"http://localhost:{TB}/?darkMode=false#scalars", wait_until="domcontentloaded")
    page.wait_for_timeout(15000)
    # enlarge the accuracy chart and collapse the learning-rate group
    page.locator("[title='Toggle fullscreen'], [aria-label='Toggle fullscreen'], paper-icon-button[icon='fullscreen']").first.click()
    page.wait_for_timeout(3000)
    page.screenshot(path=str(SHOTS / "tensorboard_scalars.png"))
    # the operation graph lives in its own run, so a second instance shows it directly
    page.goto(f"http://localhost:{TB + 1}/?darkMode=false#graphs", wait_until="domcontentloaded")
    page.wait_for_timeout(15000)
    page.screenshot(path=str(SHOTS / "tensorboard_graph.png"))
    print("saved TensorBoard screenshots")


def main() -> None:
    what = sys.argv[1:] or ["app", "tensorboard"]
    procs = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            if "app" in what:
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", "app/app.py", f"--server.port={APP}",
                     "--server.headless=true", "--browser.gatherUsageStats=false"], cwd=ROOT))
                wait(f"http://localhost:{APP}/_stcore/health")
                drive_app(browser)
            if "tensorboard" in what:
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "tensorboard.main", "--logdir", "logs", "--port", str(TB)], cwd=ROOT))
                procs.append(subprocess.Popen(
                    [sys.executable, "-m", "tensorboard.main", "--logdir", "logs/graph_baseline", "--port", str(TB + 1)], cwd=ROOT))
                wait(f"http://localhost:{TB}")
                wait(f"http://localhost:{TB + 1}")
                drive_tensorboard(browser)
            browser.close()
    finally:
        for pr in procs:
            pr.terminate()


if __name__ == "__main__":
    main()
