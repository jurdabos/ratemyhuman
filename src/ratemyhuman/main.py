"""
Legacy ``hello-world`` entry point retained for sanity checks.

The real CLI lives in :mod:`ratemyhuman.cli`; this stub is kept so that
``python -m ratemyhuman.main`` still produces a recognisable greeting.
"""


def main() -> None:
    """
    Prints a short greeting confirming the package is importable.

    Used as a smoke check during installation; the production CLI
    is exposed via :func:`ratemyhuman.cli.main`.
    """
    print("hello, ratemyhuman")


if __name__ == "__main__":
    main()
