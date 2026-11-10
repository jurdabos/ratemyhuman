"""
Tests for the CLI module.

Covers the ratemyhuman Click group wiring plus contract tests that run the
shared acidbase.push workflow through this repo's CLI entry point. Unit
tests for the underlying push helpers live upstream in acidbase
(tests/test_push.py).
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner
from PIL import Image

from ratemyhuman.cli import cli


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
            "happy": 0.90,
            "angry": 0.01,
            "disgust": 0.01,
            "fear": 0.01,
            "neutral": 0.03,
            "sad": 0.01,
            "surprise": 0.03,
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

    @patch("acidbase.push.get_project_root")
    @patch("acidbase.push._count_unpushed", return_value=0)
    @patch("acidbase.push._run")
    def test_push_dry_run(self, mock_run, mock_unpushed, mock_root, tmp_path):
        """
        Verifies dry-run mode previews without making changes.
        """
        mock_root.return_value = tmp_path
        mock_run.return_value = MagicMock(stdout="?? model.pt\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["push", "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower()

    @patch("acidbase.push.get_project_root")
    @patch("acidbase.push._count_unpushed", return_value=0)
    @patch("acidbase.push._run")
    def test_push_clean_tree(self, mock_run, mock_unpushed, mock_root, tmp_path):
        """
        Verifies early exit when the working tree is clean.
        """
        mock_root.return_value = tmp_path
        mock_run.return_value = MagicMock(stdout="")
        runner = CliRunner()
        result = runner.invoke(cli, ["push"])
        assert result.exit_code == 0
        assert "clean" in result.output.lower()

    @patch("acidbase.push.get_project_root")
    @patch("acidbase.push._count_unpushed", return_value=0)
    @patch("acidbase.push._has_changes", return_value=False)
    @patch("acidbase.push._run")
    def test_push_full_flow(self, mock_run, mock_has_changes, mock_unpushed, mock_root, tmp_path):
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
