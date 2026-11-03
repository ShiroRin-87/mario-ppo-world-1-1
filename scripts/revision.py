"""Read the source revision from a repository or a packaged snapshot."""

import shutil
import subprocess


def git_output(root, *args):
    if shutil.which("git") is None:
        return None
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_revision(root):
    snapshot = root / "SOURCE_COMMIT"
    if snapshot.is_file():
        return snapshot.read_text(encoding="utf-8").strip() or None
    return git_output(root, "rev-parse", "HEAD")
