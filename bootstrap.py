#!/usr/bin/env python3
"""
One-shot setup: virtualenv, dependencies, checkpoints, checksum verification.

    python bootstrap.py

Works on Windows, macOS and Linux. Idempotent -- an existing virtualenv and
already-downloaded checkpoints are reused, and checksums are re-verified either
way. Only the standard library is used, so it runs before anything is installed.

Exits non-zero if a download fails or a checksum does not match, rather than
leaving a corrupted checkpoint in place for the comparison to read.
"""

import hashlib
import os
import platform
import subprocess
import sys
import urllib.request

VENV_DIR = ".venv"
CHECKPOINT_DIR = "checkpoints"
REQUIREMENTS_FILE = "requirements.txt"

# (filename, url, expected sha256)
CHECKPOINTS = (
    (
        "best_model.pkl",
        "https://github.com/tanguymagne/UVDoc/raw/main/model/best_model.pkl",
        "7e90861b8a516eb4bc51f84bd889cb77275743d2d1d3ca8091951ec9f2b7da23",
    ),
    (
        "inference.pdiparams",
        "https://huggingface.co/PaddlePaddle/UVDoc/resolve/main/inference.pdiparams",
        "810488899520e0da843b9bd9769ba4949f1c81e357f0eceb12d4a7da459c3eca",
    ),
    (
        "inference.json",
        "https://huggingface.co/PaddlePaddle/UVDoc/resolve/main/inference.json",
        "2c2bc3e0f15e782cf8f2ad411b5033d99ca504fe88648f8054a5e925ba2336e0",
    ),
)

IS_WINDOWS = os.name == "nt"


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}", flush=True)


def venv_python_path(venv_dir=VENV_DIR):
    """Path to the interpreter inside a virtualenv, per platform layout."""
    if IS_WINDOWS:
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def run(command, **kwargs):
    """Run a command, raising if it fails.

    Our own stdout is flushed first: when this script's output is piped it is
    block-buffered, while the child writes to the pipe immediately, so without
    the flush the child's lines appear before the heading that introduces them.
    """
    sys.stdout.flush()
    subprocess.run(command, check=True, **kwargs)


def sha256_of(path, chunk_size=1 << 20):
    """Streaming SHA-256 so a 32 MB checkpoint is not held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_virtualenv():
    """Create the virtualenv if absent. Returns the interpreter path."""
    heading("1/3  Python environment")
    interpreter = venv_python_path()
    if os.path.exists(interpreter):
        print(f"  reusing {VENV_DIR}")
    else:
        run([sys.executable, "-m", "venv", VENV_DIR])
        print(f"  created {VENV_DIR}")
    version = subprocess.run(
        [interpreter, "--version"], capture_output=True, text=True
    ).stdout.strip()
    print(f"  {version}  ({platform.system()} {platform.machine()})")
    return interpreter


def install_dependencies(interpreter):
    """Install requirements.txt, preferring CPU-only torch where it matters."""
    heading("2/3  Dependencies")
    run([interpreter, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])

    # On Linux the default index serves CUDA-enabled torch wheels (~2.5 GB).
    # torch is only used here to read a checkpoint, so the CPU build suffices.
    if platform.system() == "Linux":
        print("  installing CPU-only torch (Linux)")
        run([
            interpreter, "-m", "pip", "install", "--quiet", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])

    run([interpreter, "-m", "pip", "install", "--quiet", "-r", REQUIREMENTS_FILE])

    report = (
        "import importlib\n"
        "for module in ('numpy', 'paddle', 'torch', 'PIL'):\n"
        "    version = getattr(importlib.import_module(module), '__version__', 'ok')\n"
        "    print(f'  {module:8s} {version}')\n"
    )
    run([interpreter, "-c", report])


def download_with_progress(url, destination):
    """Fetch `url` to `destination`, printing percentage progress."""

    def report(block_index, block_size, total_size):
        if total_size <= 0:
            return
        percent = min(100, block_index * block_size * 100 // total_size)
        print(f"\r    {percent:3d}%  {total_size / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()


def fetch_checkpoints():
    """Download and verify every checkpoint. Returns True if all are correct."""
    heading("3/3  Checkpoints (64 MB, not stored in git)")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for filename, url, expected_sha256 in CHECKPOINTS:
        destination = os.path.join(CHECKPOINT_DIR, filename)

        if os.path.exists(destination) and sha256_of(destination) == expected_sha256:
            print(f"  ok (cached)  {filename}")
            continue

        print(f"  downloading  {filename}")
        try:
            download_with_progress(url, destination)
        except Exception as error:
            print(f"  DOWNLOAD FAILED for {filename}: {error}", file=sys.stderr)
            return False

        actual_sha256 = sha256_of(destination)
        if actual_sha256 != expected_sha256:
            print(f"  CHECKSUM MISMATCH for {filename}", file=sys.stderr)
            print(f"    expected {expected_sha256}", file=sys.stderr)
            print(f"    actual   {actual_sha256}", file=sys.stderr)
            return False
        print(f"  ok           {filename}")

    return True


def main():
    """Set up the environment and print the commands to run the checks."""
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"{REQUIREMENTS_FILE} not found -- run this from the repository root.",
              file=sys.stderr)
        return 1

    interpreter = create_virtualenv()
    install_dependencies(interpreter)
    if not fetch_checkpoints():
        return 1

    heading("Ready. Run the checks:")
    print(f"  {interpreter} compare.py")
    print("      weights (bit-wise) + architecture (structural)")
    print(f"  {interpreter} verify_determinism.py")
    print("      same image -> same grid, bit for bit")
    print("\nBoth exit 0 when every check passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
