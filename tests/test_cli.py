"""Tests for the CLI module."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from ratemyhuman.cli import (
    _auto_commit_message,
    _find_dvc_changed_outs,
    _find_untracked_for_dvc,
    _get_project_root,
    _has_changes,
    _hooks_modified_files,
    cli,
    DEFAULT_SIZE_THRESHOLD,
)


class TestHooksModifiedFiles:
    """
    Tests the _hooks_modified_files helper.
    """

    def test_detects_hook_message(self):
        """
        Verifies detection of pre-commit hook modification message.
        """
        assert _hooks_modified_files("Files were modified by this hook") is True

    def test_ignores_unrelated(self):
        """
        Verifies that unrelated output is not flagged.
        """
        assert _hooks_modified_files("Committed successfully") is False

    def test_case_insensitive(self):
        """
        Verifies case-insensitive matching.
        """
        assert _hooks_modified_files("FILES WERE MODIFIED BY THIS HOOK") is True


class TestAutoCommitMessage:
    """
    Tests the _auto_commit_message helper.
    """

    def test_single_file(self):
        """
        Verifies message for a single file.
        """
        msg = _auto_commit_message(["data/model.pt"])
        assert "model.pt" in msg
        assert msg.startswith("chore:")

    def test_many_files_truncated(self):
        """
        Verifies truncation for >3 files.
        """
        files = [f"f{i}.bin" for i in range(5)]
        msg = _auto_commit_message(files)
        assert "+2 more" in msg

    def test_empty_list(self):
        """
        Verifies fallback message with no files.
        """
        assert _auto_commit_message([]) == "chore: update tracked files"


class TestGetProjectRoot:
    """
    Tests _get_project_root.
    """

    def test_finds_root_with_both_markers(self, tmp_path, monkeypatch):
        """
        Verifies detection with pyproject.toml and .dvc/.
        """
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / ".dvc").mkdir()
        sub = tmp_path / "src"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert _get_project_root() == tmp_path

    def test_finds_root_with_pyproject_only(self, tmp_path, monkeypatch):
        """
        Verifies fallback to pyproject.toml without .dvc.
        """
        (tmp_path / "pyproject.toml").touch()
        monkeypatch.chdir(tmp_path)
        assert _get_project_root() == tmp_path

    def test_fallback_to_cwd(self, tmp_path, monkeypatch):
        """
        Verifies fallback to cwd when no markers exist.
        """
        monkeypatch.chdir(tmp_path)
        assert _get_project_root() == tmp_path


class TestHasChanges:
    """
    Tests _has_changes.
    """

    @patch("ratemyhuman.cli._run")
    def test_clean(self, mock_run, tmp_path):
        """
        Verifies False for a clean working tree.
        """
        mock_run.return_value = MagicMock(stdout="")
        assert _has_changes(tmp_path) is False

    @patch("ratemyhuman.cli._run")
    def test_dirty(self, mock_run, tmp_path):
        """
        Verifies True for a dirty working tree.
        """
        mock_run.return_value = MagicMock(stdout=" M file.py\n")
        assert _has_changes(tmp_path) is True


class TestFindUntrackedForDvc:
    """
    Tests _find_untracked_for_dvc.
    """

    @patch("ratemyhuman.cli._run")
    def test_finds_binary_by_extension(self, mock_run, tmp_path):
        """
        Verifies detection of untracked binary files by extension.
        """
        mock_run.return_value = MagicMock(stdout="?? model.pt\n")
        (tmp_path / "model.pt").write_bytes(b"x" * 100)
        result = _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD)
        assert "model.pt" in result

    @patch("ratemyhuman.cli._run")
    def test_finds_large_unknown_file(self, mock_run, tmp_path):
        """
        Verifies detection of files exceeding the size threshold.
        """
        mock_run.return_value = MagicMock(stdout="?? bigfile.dat\n")
        (tmp_path / "bigfile.dat").write_bytes(b"x" * (DEFAULT_SIZE_THRESHOLD + 1))
        result = _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD)
        assert "bigfile.dat" in result

    @patch("ratemyhuman.cli._run")
    def test_skips_code_files(self, mock_run, tmp_path):
        """
        Verifies code/config files are not flagged for DVC.
        """
        mock_run.return_value = MagicMock(stdout="?? script.py\n")
        (tmp_path / "script.py").write_text("x")
        assert _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD) == []

    @patch("ratemyhuman.cli._run")
    def test_skips_dvc_pointers(self, mock_run, tmp_path):
        """
        Verifies .dvc pointer files are skipped.
        """
        mock_run.return_value = MagicMock(stdout="?? data.csv.dvc\n")
        (tmp_path / "data.csv.dvc").write_text("md5: abc")
        assert _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD) == []

    @patch("ratemyhuman.cli._run")
    def test_skips_gitignore(self, mock_run, tmp_path):
        """
        Verifies .gitignore files are skipped.
        """
        mock_run.return_value = MagicMock(stdout="?? .gitignore\n")
        (tmp_path / ".gitignore").write_text("*.log")
        assert _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD) == []

    @patch("ratemyhuman.cli._run")
    def test_skips_tracked_files(self, mock_run, tmp_path):
        """
        Verifies that tracked (non-??) files are ignored.
        """
        mock_run.return_value = MagicMock(stdout=" M model.pt\n")
        (tmp_path / "model.pt").write_bytes(b"x")
        assert _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD) == []

    @patch("ratemyhuman.cli._run")
    def test_skips_directories(self, mock_run, tmp_path):
        """
        Verifies directories are skipped.
        """
        mock_run.return_value = MagicMock(stdout="?? data/\n")
        (tmp_path / "data").mkdir()
        assert _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD) == []

    @patch("ratemyhuman.cli._run")
    def test_empty_status(self, mock_run, tmp_path):
        """
        Verifies empty result when no untracked files exist.
        """
        mock_run.return_value = MagicMock(stdout="")
        assert _find_untracked_for_dvc(tmp_path, DEFAULT_SIZE_THRESHOLD) == []


class TestFindDvcChangedOuts:
    """
    Tests _find_dvc_changed_outs.
    """

    @patch("ratemyhuman.cli._run")
    def test_finds_modified(self, mock_run, tmp_path):
        """
        Verifies detection of modified DVC outputs.
        """
        mock_run.return_value = MagicMock(
            stdout="file.png.dvc:\n\tmodified: file.png\n"
        )
        result = _find_dvc_changed_outs(tmp_path)
        assert "file.png" in result

    @patch("ratemyhuman.cli._run")
    def test_no_changes_empty(self, mock_run, tmp_path):
        """
        Verifies empty result for empty dvc status.
        """
        mock_run.return_value = MagicMock(stdout="")
        assert _find_dvc_changed_outs(tmp_path) == []

    @patch("ratemyhuman.cli._run")
    def test_no_changes_explicit(self, mock_run, tmp_path):
        """
        Verifies empty result when 'no changes' is reported.
        """
        mock_run.return_value = MagicMock(stdout="No changes.")
        assert _find_dvc_changed_outs(tmp_path) == []


class TestCliGroup:
    """
    Tests the top-level CLI group.
    """

    def test_help(self):
        """
        Verifies CLI help output.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "RateMyHuman" in result.output

    def test_verbose_flag(self):
        """
        Verifies --verbose flag is accepted.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ["-v", "--help"])
        assert result.exit_code == 0


class TestClassifyCommand:
    """
    Tests the classify CLI command.
    """

    def _make_mock_result(self):
        """
        Creates a mock ValenceResult for testing.
        """
        r = MagicMock()
        r.label = "Positive"
        r.confidence = 0.95
        r.emotion_scores = {
            "happy": 0.90, "angry": 0.01, "disgust": 0.01,
            "fear": 0.01, "neutral": 0.03, "sad": 0.01, "surprise": 0.03,
        }
        r.valence_scores = {"Positive": 0.93, "Negative": 0.04, "Neutral": 0.03}
        r.__str__ = MagicMock(return_value="Valence: Positive (95.0%)")
        return r

    @patch("ratemyhuman.model.ValenceDetector")
    def test_classify_plain(self, MockDetector, tmp_path):
        """
        Verifies plain-text classify output.
        """
        img = tmp_path / "face.png"
        Image.new("RGB", (48, 48)).save(img)
        MockDetector.return_value.classify.return_value = self._make_mock_result()
        runner = CliRunner()
        result = runner.invoke(cli, ["classify", str(img)])
        assert result.exit_code == 0

    @patch("ratemyhuman.model.ValenceDetector")
    def test_classify_json(self, MockDetector, tmp_path):
        """
        Verifies JSON classify output.
        """
        img = tmp_path / "face.png"
        Image.new("RGB", (48, 48)).save(img)
        MockDetector.return_value.classify.return_value = self._make_mock_result()
        runner = CliRunner()
        result = runner.invoke(cli, ["classify", str(img), "--json"])
        assert result.exit_code == 0
        assert "Positive" in result.output

    @patch("ratemyhuman.model.ValenceDetector")
    def test_classify_error(self, MockDetector, tmp_path):
        """
        Verifies error handling when detection fails.
        """
        img = tmp_path / "face.png"
        Image.new("RGB", (48, 48)).save(img)
        MockDetector.return_value.classify.side_effect = ValueError("No face detected")
        runner = CliRunner()
        result = runner.invoke(cli, ["classify", str(img)])
        assert result.exit_code != 0


class TestExploreCommand:
    """
    Tests the explore CLI command.
    """

    @patch("ratemyhuman.explore.run_exploration")
    def test_explore_default(self, mock_explore):
        """
        Verifies the explore command invokes run_exploration.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ["explore"])
        assert result.exit_code == 0
        mock_explore.assert_called_once()


class TestValidateCommand:
    """
    Tests the validate CLI command.
    """

    @patch("ratemyhuman.validate.run_validation")
    def test_validate_default(self, mock_validate):
        """
        Verifies the validate command invokes run_validation.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code == 0
        mock_validate.assert_called_once()


class TestDemoCommand:
    """
    Tests the demo CLI command.
    """

    @patch("ratemyhuman.app.launch")
    def test_demo_default(self, mock_launch):
        """
        Verifies the demo command invokes launch.
        """
        runner = CliRunner()
        result = runner.invoke(cli, ["demo"])
        assert result.exit_code == 0
        mock_launch.assert_called_once()


class TestPushCommand:
    """
    Tests the push CLI command.
    """

    @patch("ratemyhuman.cli._get_project_root")
    @patch("ratemyhuman.cli._find_untracked_for_dvc")
    @patch("ratemyhuman.cli._find_dvc_changed_outs", return_value=[])
    @patch("ratemyhuman.cli._run")
    def test_push_dry_run(self, mock_run, mock_changed, mock_untracked, mock_root, tmp_path):
        """
        Verifies dry-run mode previews without making changes.
        """
        mock_root.return_value = tmp_path
        (tmp_path / "model.pt").write_bytes(b"x" * 1000)
        mock_untracked.return_value = ["model.pt"]
        mock_run.return_value = MagicMock(stdout="?? model.pt\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["push", "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    @patch("ratemyhuman.cli._get_project_root")
    @patch("ratemyhuman.cli._find_untracked_for_dvc", return_value=[])
    @patch("ratemyhuman.cli._find_dvc_changed_outs", return_value=[])
    @patch("ratemyhuman.cli._run")
    def test_push_clean_tree(self, mock_run, mock_changed, mock_untracked, mock_root, tmp_path):
        """
        Verifies early exit when the working tree is clean.
        """
        mock_root.return_value = tmp_path
        mock_run.return_value = MagicMock(stdout="")
        runner = CliRunner()
        result = runner.invoke(cli, ["push"])
        assert result.exit_code == 0
        assert "clean" in result.output.lower()

    @patch("ratemyhuman.cli._get_project_root")
    @patch("ratemyhuman.cli._find_untracked_for_dvc", return_value=[])
    @patch("ratemyhuman.cli._find_dvc_changed_outs", return_value=[])
    @patch("ratemyhuman.cli._has_changes", return_value=False)
    @patch("ratemyhuman.cli._run")
    def test_push_full_flow(self, mock_run, mock_has_changes, mock_changed, mock_untracked, mock_root, tmp_path):
        """
        Verifies the full push flow with commit and push.
        """
        mock_root.return_value = tmp_path

        def run_effect(cmd, cwd, check=True):
            """
            Simulates subprocess responses for the push workflow.
            """
            r = MagicMock()
            if cmd[:2] == ["git", "status"]:
                r.stdout = " M model.py\n"
            elif cmd[:2] == ["git", "diff"]:
                r.stdout = "model.py\n"
            else:
                r.stdout = ""
            return r

        mock_run.side_effect = run_effect
        runner = CliRunner()
        result = runner.invoke(cli, ["push", "-m", "test commit"])
        assert result.exit_code == 0
        assert "done" in result.output.lower()
