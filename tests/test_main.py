"""Smoke test for the legacy ratemyhuman.main entry point."""
from ratemyhuman.main import main


def test_main_prints_greeting(capsys):
    """
    Verifies that main() prints the expected greeting to stdout.

    Acts as a sanity check that the package is importable and the
    legacy hello-world entry point still works.
    """
    main()
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello, ratemyhuman"
