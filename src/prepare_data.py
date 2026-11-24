"""Download a reproducible PlantVillage subset and split it without leaf leakage.

The full archive is 2.2 GB, so this script opens the remote ZIP with HTTP range
requests and fetches only the members it needs.

PlantVillage contains several photographs of the same physical leaf. Images are
therefore sampled and split by *leaf group* (using the authors' leaf-map.json)
so the same leaf can never appear in both the training and the test set.

  python -m src.prepare_data            reproduce the committed data/manifest.csv
  python -m src.prepare_data --rebuild  sample again from scratch (same seed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src.config import CLASS_DIRS, DATA_DIR, MANIFEST, PER_CLASS, ROOT, SEED, SPLIT

REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"   # pinned dataset commit
BASE = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{REVISION}"
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


_cdn: dict[str, str] = {}
_cdn_lock = threading.Lock()


def _resolve(url: str, refresh: bool = False) -> str:
    """Follow the hub redirect once and reuse the signed CDN link afterwards."""
    with _cdn_lock:
        if refresh:
            _cdn.pop(url, None)
        if url not in _cdn:
            r = _session().head(url, allow_redirects=True, timeout=60)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  rate limited, waiting {wait}s", flush=True)
                time.sleep(wait)
                r = _session().head(url, allow_redirects=True, timeout=60)
            r.raise_for_status()
            _cdn[url] = r.url
        return _cdn[url]


def _get(url: str, headers=None, tries: int = 8) -> requests.Response:
    err = ""
    for i in range(tries):
        try:
            r = _session().get(_resolve(url, refresh=i > 0 and "403" in err),
                               headers=headers, timeout=60)
            if r.status_code in (200, 206):
                return r
            err = f"HTTP {r.status_code}"
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 60)))
        except requests.RequestException as e:
            err = type(e).__name__
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"failed after {tries} tries: {url} ({err})")


class RemoteFile:
    """Minimal seekable file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        r = _get(url, {"Range": "bytes=0-0"})
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self._start, self._buf = 0, b""

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        lo = self.pos - self._start
        if not (0 <= lo and lo + n <= len(self._buf)):
            end = min(self.size, self.pos + max(n, 1 << 20)) - 1
            self._buf = _get(self.url, {"Range": f"bytes={self.pos}-{end}"}).content
            self._start, lo = self.pos, 0
        out = self._buf[lo:lo + n]
        self.pos += n
        return out


def fetch_member(info: zipfile.ZipInfo, zip_url: str) -> bytes:
    """Read one ZIP member with a single range request and verify its CRC."""
    end = info.header_offset + 30 + len(info.filename.encode()) + 512 + info.compress_size
    raw = _get(zip_url, {"Range": f"bytes={info.header_offset}-{end}"}).content
    n, e = struct.unpack("<HH", raw[26:30])
    body = raw[30 + n + e: 30 + n + e + info.compress_size]
    data = zlib.decompress(body, -15) if info.compress_type == 8 else body
    if zlib.crc32(data) != info.CRC:
        raise RuntimeError(f"CRC mismatch: {info.filename}")
    return data


def build_manifest(names: list[str]) -> pd.DataFrame:
    """Choose images leaf-by-leaf, then assign whole leaves to a split."""
    leaf_map = json.loads(_get(f"{BASE}/leaf_grouping/leaf-map.json").content)
    rows = []
    for cls in CLASS_DIRS:
        rng = random.Random(f"{SEED}-{cls}")
        groups: dict[str, list[str]] = {}
        for path in sorted(p for p in names if p.startswith(f"raw/color/{cls}/")):
            key = path.rsplit("/", 1)[1].split("___", 1)[-1].rsplit(".", 1)[0].lower()
            hit = [v for v in leaf_map.get(key, []) if v.startswith(cls + ":::")]
            gid = hit[0].split(":::")[1] if hit else "single:" + key
            groups.setdefault(gid, []).append(path)
        order = sorted(groups)
        rng.shuffle(order)
        chosen, total = [], 0
        for gid in order:
            take = groups[gid][: PER_CLASS - total]
            if take:
                chosen.append((gid, take))
                total += len(take)
            if total >= PER_CLASS:
                break
        # Fill test, then validation, then train - always whole leaves.
        want = {"test": round(total * SPLIT[2]), "val": round(total * SPLIT[1])}
        have = {"test": 0, "val": 0}
        for gid, paths in chosen:
            split = "train"
            for s in ("test", "val"):
                if have[s] + len(paths) <= want[s]:
                    split = s
                    have[s] += len(paths)
                    break
            for p in paths:
                fn = p.rsplit("/", 1)[1].split("___", 1)[0] + ".jpg"
                rows.append(dict(source_path=p, class_dir=cls, leaf_id=gid,
                                 split=split, filename=fn))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    zip_url = f"{BASE}/data.zip"
    print("reading remote ZIP index ...", flush=True)
    zf = zipfile.ZipFile(RemoteFile(zip_url))
    infos = {i.filename: i for i in zf.infolist() if i.filename.startswith("raw/color/") and not i.is_dir()}
    print(f"{len(infos)} colour images available", flush=True)

    if args.rebuild or not MANIFEST.exists():
        df = build_manifest(list(infos))
        df.to_csv(MANIFEST, index=False)
    else:
        df = pd.read_csv(MANIFEST, dtype={"leaf_id": str})

    def job(row) -> str:
        out = DATA_DIR / row.class_dir / row.filename
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(fetch_member(infos[row.source_path], zip_url))
            tmp.replace(out)
        return hashlib.sha256(out.read_bytes()).hexdigest()[:16]

    with ThreadPoolExecutor(6) as ex:
        hashes = []
        for i, h in enumerate(ex.map(job, df.itertuples()), 1):
            hashes.append(h)
            if i % 250 == 0:
                print(f"  {i}/{len(df)} images ready", flush=True)
    if "sha256_16" in df:
        bad = int((df["sha256_16"] != pd.Series(hashes, index=df.index)).sum())
        if bad:
            raise SystemExit(f"{bad} files do not match the manifest checksums")
        print("all files match the manifest checksums")
    df["sha256_16"] = hashes
    df.to_csv(MANIFEST, index=False)
    print(pd.crosstab(df.class_dir, df.split, margins=True))
    leak = df.groupby(["class_dir", "leaf_id"]).split.nunique().max()
    print("max splits per leaf (must be 1):", leak, "->", MANIFEST.relative_to(ROOT))


if __name__ == "__main__":
    main()
