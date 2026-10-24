"""Download the blue-noise basis used by the film-grain operator.

The pipeline samples film grain from a pre-computed 2048x2048x5 blue-noise
texture (``data/blue_noise/blue_noise_2048_5ch.npy``, ~40 MB). The file is
published as a GitHub Release asset rather than stored in git.

    python scripts/download_blue_noise.py            # download + verify
    python scripts/download_blue_noise.py --verify   # only check an existing file

Any other zero-mean blue-noise texture of shape HxW or HxWxC can be used
instead (set ``blue_noise_path`` of ``FilmGrainStep`` in the config); the
checksum below only identifies the basis used for the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/TytanMikJas/AbsoluteDegradation/releases/download/"
    "v1.0.0/blue_noise_2048_5ch.npy"
)
SHA256 = "c6b1d40b4076b3c47dca86af88b24d74c0b04e8ac9b187dfba46bfd13d6e00fb"
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "blue_noise"
    / "blue_noise_2048_5ch.npy"
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--verify", action="store_true", help="Only verify an existing file."
    )
    args = parser.parse_args()

    if not args.verify:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".part")
        print(f"Downloading {args.url}\n         -> {args.out}")
        urllib.request.urlretrieve(args.url, tmp)
        tmp.replace(args.out)

    if not args.out.exists():
        print(f"File not found: {args.out}", file=sys.stderr)
        return 1

    digest = sha256sum(args.out)
    if digest != SHA256:
        print(
            f"SHA-256 mismatch for {args.out}\n  expected {SHA256}\n  got      {digest}",
            file=sys.stderr,
        )
        return 1
    print(f"OK  {args.out}  (sha256 {digest[:16]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
