"""
Command-line interface for ratemyhuman.

Provides subcommands for classification, exploration, validation,
and the DVC + git push workflow.
"""
import warnings
warnings.filterwarnings("ignore", message="torch._dynamo.allow_in_graph is deprecated")
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import click


# Byte threshold above which an untracked file is auto-added to DVC
DEFAULT_SIZE_THRESHOLD = 1_048_576  # 1 MB

# Extensions always routed through DVC regardless of size
DVC_EXTENSIONS: set[str] = {
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".heic", ".heif", ".svg", ".ico", ".raw", ".cr2", ".nef", ".arw",
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".ts", ".flv", ".wmv",
    # Audio
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma",
    # ML artefacts
    ".h5", ".hdf5", ".pkl", ".pickle", ".pt", ".pth", ".onnx",
    ".safetensors", ".bin", ".npy", ".npz",
    # Archives & data
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".csv", ".parquet", ".feather", ".arrow",
    ".db", ".sqlite", ".sqlite3",
}

# Extensions that are clearly source/config and should stay in git only
_CODE_EXTENSIONS: set[str] = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".cfg", ".ini", ".sh", ".ps1", ".bat", ".cmd",
    ".html", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".lock", ".gitignore", ".gitattributes", ".dvcignore",
}


def _get_project_root() -> Path:
    """
    Locates the project root by walking up to find pyproject.toml + .dvc/.

    Falls back to the nearest pyproject.toml ancestor, or the current
    working directory if no marker is found.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / ".dvc").is_dir():
            return parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """
    Runs a subprocess command, returning CompletedProcess.

    Captures stdout/stderr as UTF-8 text so callers can parse them; raises
    ``CalledProcessError`` when ``check`` is True and the command fails.
    """
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=check,
    )


def _has_changes(root: Path) -> bool:
    """
    Checks whether the working tree has any staged or unstaged changes.

    Used after commits to detect post-commit hooks that left the tree dirty.
    """
    result = _run(["git", "status", "--porcelain"], cwd=root, check=False)
    return bool(result.stdout.strip())


def _hooks_modified_files(output: str) -> bool:
    """
    Checks whether pre-commit hooks modified files (retryable failure).

    Inspects the captured commit output for the standard pre-commit warning
    so the caller can re-stage and retry instead of aborting.
    """
    return "files were modified by this hook" in output.lower()


def _find_untracked_for_dvc(root: Path, size_threshold: int) -> list[str]:
    """
    Finds untracked files that should be DVC-tracked.

    Matching by known binary/data extension or by exceeding the size threshold.
    """
    result = _run(["git", "status", "--porcelain"], cwd=root, check=False)
    candidates: list[str] = []
    for line in result.stdout.strip().splitlines():
        if not line.startswith("?? "):
            continue
        rel_path = line[3:].strip().strip('"')
        full_path = root / rel_path
        if not full_path.is_file():
            continue
        # Skipping DVC pointers and gitignore files
        if rel_path.endswith(".dvc") or rel_path.endswith(".gitignore"):
            continue
        suffix = full_path.suffix.lower()
        # Skipping known source/config files
        if suffix in _CODE_EXTENSIONS:
            continue
        # Adding if extension matches DVC set or file exceeds size threshold
        if suffix in DVC_EXTENSIONS or full_path.stat().st_size >= size_threshold:
            candidates.append(rel_path)
    return candidates


def _find_dvc_changed_outs(root: Path) -> list[str]:
    """
    Finds DVC-tracked outputs that have been modified since last ``dvc add``.

    Parses ``dvc status`` text output for 'modified:' lines.
    """
    result = _run(["dvc", "status"], cwd=root, check=False)
    if not result.stdout.strip() or "no changes" in result.stdout.lower():
        return []
    changed: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("modified:"):
            path = stripped.split(":", 1)[1].strip()
            changed.append(path)
    return changed


def _auto_commit_message(dvc_files: list[str]) -> str:
    """
    Generates an automatic commit message from DVC-added file names.

    Uses up to three file names verbatim and summarises the rest as
    ``+N more``; falls back to a generic message when no files are given.
    """
    if dvc_files:
        names = ", ".join(Path(f).name for f in dvc_files[:3])
        suffix = f" (+{len(dvc_files) - 3} more)" if len(dvc_files) > 3 else ""
        return f"chore: ingest {names}{suffix}"
    return "chore: update tracked files"


@click.group(context_settings={"max_content_width": shutil.get_terminal_size().columns})
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """
    RateMyHuman: facial valence detection CLI.

    Top-level command group exposing classify, explore, validate, demo,
    and push subcommands; the ``--verbose`` flag enables debug logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# -------------------------------------------------------------------
# classify
# -------------------------------------------------------------------
@cli.command("classify")
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("--device", default=None, help="Force device (cuda / cpu)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def classify_cmd(image: str, device: str | None, as_json: bool) -> None:
    """
    Classifies the valence of a face in IMAGE.

    Runs the full pipeline: face detection -> emotion inference -> valence mapping.
    """
    from ratemyhuman.model import ValenceDetector
    try:
        detector = ValenceDetector(device=device)
        result = detector.classify(image)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        raise SystemExit(1)
    if as_json:
        import json
        click.echo(json.dumps({
            "label": result.label,
            "confidence": round(result.confidence, 4),
            "emotion_scores": {k: round(v, 4) for k, v in result.emotion_scores.items()},
            "valence_scores": {k: round(v, 4) for k, v in result.valence_scores.items()},
        }, indent=2))
    else:
        click.echo(result)
        click.echo(f"  Emotions: { {k: round(v, 4) for k, v in result.emotion_scores.items()} }")
        click.echo(f"  Valence:  { {k: round(v, 4) for k, v in result.valence_scores.items()} }")


# -------------------------------------------------------------------
# explore
# -------------------------------------------------------------------
@cli.command("explore")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Dataset root (default: data/)")
@click.option("--output-dir", type=click.Path(file_okay=False), default=None,
              help="Output directory for plots (default: docs/)")
def explore_cmd(data_dir: str | None, output_dir: str | None) -> None:
    """
    Explores the FER2013 dataset.

    Produces class/valence distribution stats, sample grids, and integrity checks.
    """
    from ratemyhuman.explore import run_exploration
    run_exploration(
        data_dir=Path(data_dir) if data_dir else None,
        output_dir=Path(output_dir) if output_dir else None,
    )
    click.echo(click.style("\n✓ Exploration complete.", fg="green"))


# -------------------------------------------------------------------
# validate
# -------------------------------------------------------------------
@cli.command("validate")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Test split directory (default: data/test/)")
@click.option("--output-dir", type=click.Path(file_okay=False), default=None,
              help="Output directory for plots (default: docs/)")
@click.option("--split", default="test", help="Dataset split to validate on")
def validate_cmd(data_dir: str | None, output_dir: str | None, split: str) -> None:
    """
    Validates the pipeline on labelled data.

    Runs batch inference on all images in the split, computes metrics
    (accuracy, F1, MCC, confusion matrix), and saves plots.
    """
    from ratemyhuman.validate import run_validation
    run_validation(
        data_dir=Path(data_dir) if data_dir else None,
        output_dir=Path(output_dir) if output_dir else None,
        split=split,
    )
    click.echo(click.style("\n✓ Validation complete.", fg="green"))


# -------------------------------------------------------------------
# demo (Gradio web UI)
# -------------------------------------------------------------------
@cli.command("demo")
@click.option("--share", is_flag=True, help="Create a public Gradio link")
@click.option("--port", default=7860, type=int, show_default=True, help="Server port")
def demo_cmd(share: bool, port: int) -> None:
    """
    Launches the Gradio web UI for interactive valence detection.

    Upload a face image (or use your webcam) and get instant
    valence + emotion predictions.
    """
    from ratemyhuman.app import launch
    launch(share=share, server_port=port)


# -------------------------------------------------------------------
# push (DVC + git workflow)
# -------------------------------------------------------------------
@cli.command("push")
@click.option("--message", "-m", default=None, help="Custom commit message")
@click.option("--dry-run", is_flag=True, help="Preview without making changes")
@click.option(
    "--size-threshold",
    default=DEFAULT_SIZE_THRESHOLD,
    type=int,
    show_default=True,
    help="Min file size in bytes for auto-DVC tracking",
)
def push(message: str | None, dry_run: bool, size_threshold: int) -> None:
    """
    Ingests new big files via DVC, commits, and pushes everything.

    Automates the full workflow:
    1. Detects untracked data/binary files and runs ``dvc add``
    2. Detects modified DVC-tracked files and re-adds them
    3. Stages all changes with ``git add .``
    4. Commits with pre-commit hook retry (up to 3 attempts)
    5. Amends if post-commit hooks leave dirty state
    6. ``dvc push`` to remote storage
    7. ``git push`` to GitHub
    """
    root = _get_project_root()
    click.echo(click.style("\n=== ratemyhuman push ===", fg="cyan", bold=True))
    click.echo(f"Root: {root}\n")
    all_dvc_files: list[str] = []
    # Step 1: Finding untracked files that should go through DVC
    new_for_dvc = _find_untracked_for_dvc(root, size_threshold)
    if new_for_dvc:
        click.echo(click.style(f"① {len(new_for_dvc)} new file(s) to DVC-track:", bold=True))
        for f in new_for_dvc:
            size_mb = (root / f).stat().st_size / 1_048_576
            click.echo(f"   {f}  ({size_mb:.1f} MB)")
        if not dry_run:
            for f in new_for_dvc:
                _run(["dvc", "add", f], cwd=root)
                click.echo(f"   ✓ dvc add {f}")
        all_dvc_files.extend(new_for_dvc)
    else:
        click.echo("① No new files to DVC-track")
    # Step 2: Finding modified DVC-tracked outputs that need re-adding
    changed_outs = _find_dvc_changed_outs(root)
    if changed_outs:
        click.echo(click.style(f"\n② {len(changed_outs)} modified DVC output(s) to re-add:", bold=True))
        for f in changed_outs:
            click.echo(f"   {f}")
        if not dry_run:
            for f in changed_outs:
                _run(["dvc", "add", f], cwd=root)
                click.echo(f"   ✓ dvc add {f}")
        all_dvc_files.extend(changed_outs)
    else:
        click.echo("② No modified DVC outputs")
    # Step 3: Summarising all git changes (including newly created .dvc pointers)
    status_result = _run(["git", "status", "--porcelain"], cwd=root, check=False)
    status_lines = [ln for ln in status_result.stdout.strip().splitlines() if ln.strip()]
    if not status_lines:
        click.echo(click.style("\n✓ Nothing to push — working tree is clean.", fg="green"))
        return
    click.echo(click.style(f"\n③ {len(status_lines)} git change(s) detected:", bold=True))
    for ln in status_lines:
        click.echo(f"   {ln}")
    if dry_run:
        click.echo(click.style("\n── dry run ── no changes made", fg="yellow"))
        return
    # Step 4: Staging all changes
    click.echo(click.style("\n④ Staging all changes ...", bold=True))
    _run(["git", "add", "."], cwd=root)
    staged = _run(["git", "diff", "--cached", "--name-only"], cwd=root, check=False)
    if not staged.stdout.strip():
        click.echo(click.style("   ⚠ Nothing staged after git add — skipping commit.", fg="yellow"))
    else:
        staged_count = len(staged.stdout.strip().splitlines())
        click.echo(f"   {staged_count} file(s) staged")
        # Step 5: Committing with pre-commit hook retry
        commit_msg = message or _auto_commit_message(all_dvc_files)
        full_msg = f"{commit_msg}\n\nCo-Authored-By: Warp <agent@warp.dev>"
        click.echo(click.style("\n⑤ Committing ...", bold=True))
        max_attempts = 3
        committed = False
        for attempt in range(1, max_attempts + 1):
            try:
                _run(["git", "commit", "-m", full_msg], cwd=root)
                label = f" (attempt {attempt})" if attempt > 1 else ""
                click.echo(f"   ✓ Committed{label}")
                committed = True
                break
            except subprocess.CalledProcessError as exc:
                combined = exc.stdout + exc.stderr
                if _hooks_modified_files(combined) and attempt < max_attempts:
                    click.echo(f"   ⟳ Pre-commit hooks modified files (attempt {attempt}) — re-staging ...")
                    _run(["git", "add", "."], cwd=root)
                    continue
                click.echo(
                    click.style(f"   ✗ Commit failed (attempt {attempt}): {exc.stderr.strip()}", fg="red"),
                    err=True,
                )
                raise SystemExit(1)
        # Handling post-commit hooks that leave dirty state
        if committed and _has_changes(root):
            click.echo("   ⟳ Post-commit hook left changes — amending ...")
            _run(["git", "add", "."], cwd=root)
            _run(["git", "commit", "--amend", "--no-edit", "--no-verify"], cwd=root, check=False)
            click.echo("   ✓ Amended")
    # Step 6: DVC push
    click.echo(click.style("\n⑥ DVC push ...", bold=True))
    try:
        _run(["dvc", "push"], cwd=root)
        click.echo("   ✓ Pushed to DVC remote")
    except subprocess.CalledProcessError as exc:
        click.echo(click.style(f"   ✗ dvc push failed: {exc.stderr.strip()}", fg="red"), err=True)
        raise SystemExit(1)
    # Step 7: Git push
    click.echo(click.style("\n⑦ Pushing to GitHub ...", bold=True))
    try:
        _run(["git", "push"], cwd=root)
        click.echo("   ✓ Pushed to GitHub")
    except subprocess.CalledProcessError as exc:
        click.echo(click.style(f"   ✗ git push failed: {exc.stderr.strip()}", fg="red"), err=True)
        raise SystemExit(1)
    click.echo(click.style("\n✓ All done — changes committed & pushed.", fg="green", bold=True))


def main() -> None:
    """
    Entry point for the ratemyhuman CLI.

    Thin wrapper around the Click command group so the project's
    ``[project.scripts]`` entry can target a single callable.
    """
    cli()
