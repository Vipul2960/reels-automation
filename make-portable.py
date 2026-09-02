"""Assemble a self-contained copy that runs on a PC with nothing installed.

    python make-portable.py [destination]

Copies the code, the virtualenv, the vision models and ffmpeg/ffprobe into one
folder. On the target machine there is nothing to install: open the folder and
double-click make-reels.bat.

Deliberately NOT copied: work/ and output/ (gigabytes of downloaded video and
finished reels, all reproducible) and .anthropic_key (a secret).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CODE = ["run.py", "get_models.py", "config.json", "README.md",
        "make-reels.bat", "requirements.txt", "LICENSE"]
TREES = ["reels", "assets", "venv"]
# Applied at EVERY level of the copied trees, so keep it to names that are
# never a real package. "output" was here once and silently ate
# site-packages/scenedetect/output/, which made scene detection fail and every
# reel come out as a single uncut scene. work/ and output/ are not inside the
# copied trees anyway - they are removed separately below.
SKIP_DIRS = {"__pycache__", ".git"}


def find_binary(name: str) -> Path | None:
    """Resolve a tool to its real executable, following WinGet's link shims."""
    found = shutil.which(name)
    if not found:
        return None
    path = Path(found)
    if path.stat().st_size > 1_000_000:          # a real binary, not a shim
        return path

    # WinGet installs a tiny forwarder in Links/; hunt down the real one.
    packages = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if packages.is_dir():
        for candidate in packages.rglob(f"{name}.exe"):
            if candidate.stat().st_size > 1_000_000:
                return candidate
    return path


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def tree_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "reels-portable"
    dest = dest.resolve()
    if dest == ROOT:
        print("Destination must be a different folder.")
        return 1

    print(f"Building portable copy in:\n  {dest}\n")
    dest.mkdir(parents=True, exist_ok=True)

    for name in CODE:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, dest / name)
            print(f"  {name}")

    for name in TREES:
        src = ROOT / name
        if not src.is_dir():
            continue
        print(f"  {name}/ ... ", end="", flush=True)
        shutil.copytree(src, dest / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.pyc"))
        print(human(tree_size(dest / name)))

    # ffmpeg + ffprobe travel with the copy; util.tool() looks in bin/ first.
    bin_dir = dest / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool_name in ("ffmpeg", "ffprobe"):
        found = find_binary(tool_name)
        if found is None:
            print(f"  bin/{tool_name}.exe  NOT FOUND - install ffmpeg first")
            continue
        shutil.copy2(found, bin_dir / f"{tool_name}.exe")
        print(f"  bin/{tool_name}.exe  {human((bin_dir / f'{tool_name}.exe').stat().st_size)}")

    for junk in ("work", "output", ".anthropic_key"):
        target = dest / junk
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()

    # A virtualenv records an absolute path to the Python that made it. Dropping
    # the config makes the launcher use venv\Scripts\python.exe directly, which
    # is what make-reels.bat calls, so the copy works from any folder.
    cfg = dest / "venv" / "pyvenv.cfg"
    if cfg.exists():
        cfg.write_text(cfg.read_text(encoding="utf-8").replace("\\", "/"),
                       encoding="utf-8")

    total = tree_size(dest)
    print(f"\nDone - {human(total)} total.")
    print("Copy the whole folder to the other PC and double-click make-reels.bat.")

    print("\nVerifying the copy can start ...")
    probe = subprocess.run(
        [str(dest / "venv" / "Scripts" / "python.exe"), "-c",
         "import sys; sys.path.insert(0,'.');"
         "from reels.util import check_tools;"
         "import cv2, numpy, mediapipe;"
         "print('imports OK, missing tools:', check_tools() or 'none')"],
        cwd=dest, capture_output=True, text=True)
    print("  " + (probe.stdout.strip() or probe.stderr.strip()[-300:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
