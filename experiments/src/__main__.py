"""Allow ``python -m sdg_report_experiments`` to reproduce the results."""

from .runner import main


if __name__ == "__main__":
    raise SystemExit(main())

