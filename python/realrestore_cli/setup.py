"""Setup script for RealRestore CLI Python environment."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def check_python_version() -> bool:
    """Ensure Python 3.12+."""
    return sys.version_info >= (3, 12)


def install_patched_diffusers() -> bool:
    """Install the patched diffusers from upstream-realrestorer."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    diffusers_dir = repo_root / "upstream-realrestorer" / "diffusers"

    if not diffusers_dir.exists():
        print("Error: upstream-realrestorer/diffusers not found", file=sys.stderr)
        return False

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(diffusers_dir)],
        capture_output=True, text=True
    )
    return result.returncode == 0


def install_requirements() -> bool:
    """Install Python dependencies."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    req_file = repo_root / "python" / "requirements.txt"

    if not req_file.exists():
        print("Warning: requirements.txt not found", file=sys.stderr)
        return True

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
        capture_output=True, text=True
    )
    return result.returncode == 0


def download_model() -> bool:
    """Download model weights from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
        model_path = os.environ.get("REALRESTORE_MODEL_PATH")
        if model_path and Path(model_path).exists():
            print("Model already downloaded.")
            return True

        snapshot_download(
            repo_id="RealRestorer/RealRestorer",
            local_dir=None,  # Use HF cache
        )
        return True
    except ImportError:
        print("huggingface_hub not installed, skipping model download", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Model download failed: {e}", file=sys.stderr)
        return False


def verify_installation() -> bool:
    """Verify the environment works."""
    try:
        from diffusers import RealRestorerPipeline
        return True
    except ImportError:
        return False


def main() -> None:
    steps = {
        "python_version": check_python_version(),
    }

    if not steps["python_version"]:
        result = {"status": "error", "message": "Python 3.12+ required", "steps": steps}
        print(json.dumps(result))
        sys.exit(2)

    steps["diffusers"] = install_patched_diffusers()
    steps["requirements"] = install_requirements()
    steps["model"] = download_model()
    steps["verified"] = verify_installation()

    status = "success" if all(steps.values()) else "partial"
    result = {"status": status, "steps": steps}
    print(json.dumps(result))

    if status != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
