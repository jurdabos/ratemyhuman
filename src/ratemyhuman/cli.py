"""
Command-line interface for ratemyhuman.

Provides subcommands for classification, exploration, validation,
and the ``push`` workflow (shared from :mod:`acidbase.push`).
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message="torch._dynamo.allow_in_graph is deprecated")

import logging
import shutil
from pathlib import Path

import click

from acidbase.push import push_command


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


cli.add_command(push_command)


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
        raise SystemExit(1) from exc
    if as_json:
        import json

        click.echo(
            json.dumps(
                {
                    "label": result.label,
                    "confidence": round(result.confidence, 4),
                    "emotion_scores": {k: round(v, 4) for k, v in result.emotion_scores.items()},
                    "valence_scores": {k: round(v, 4) for k, v in result.valence_scores.items()},
                },
                indent=2,
            )
        )
    else:
        click.echo(result)
        click.echo(f"  Emotions: {  {k: round(v, 4) for k, v in result.emotion_scores.items()} }")
        click.echo(f"  Valence:  {  {k: round(v, 4) for k, v in result.valence_scores.items()} }")


# -------------------------------------------------------------------
# explore
# -------------------------------------------------------------------
@cli.command("explore")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Dataset root (default: data/)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for plots (default: docs/)",
)
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
    click.echo(click.style("\n\u2713 Exploration complete.", fg="green"))


# -------------------------------------------------------------------
# validate
# -------------------------------------------------------------------
@cli.command("validate")
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Test split directory (default: data/test/)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Output directory for plots (default: docs/)",
)
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
    click.echo(click.style("\n\u2713 Validation complete.", fg="green"))


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


def main() -> None:
    """
    Entry point for the ratemyhuman CLI.

    Thin wrapper around the Click command group so the project's
    ``[project.scripts]`` entry can target a single callable.
    """
    cli()
