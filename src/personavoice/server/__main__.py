"""CLI entrypoint: `python -m personavoice.server [--check] [--backend mac|cuda]`."""

from __future__ import annotations

import argparse
import sys

from .check import CheckReport, run_check
from .config import ConfigError, Settings

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _c(text: str, color: str, *, use_color: bool) -> str:
    return f"{color}{text}{_RESET}" if use_color else text


def _print_report(report: CheckReport, settings: Settings, *, use_color: bool) -> None:
    print(f"Persona-Voice config check  (backend={settings.backend})")
    print(f"{_DIM if use_color else ''}config: {settings.config_dir}{_RESET if use_color else ''}")
    print()

    if report.errors:
        for err in report.errors:
            print(_c("ERROR", _RED, use_color=use_color), err)
        print()

    if report.adapter_results:
        print("Adapters:")
        for r in report.adapter_results:
            mark = (
                _c("ok", _GREEN, use_color=use_color)
                if r.ok
                else _c("FAIL", _RED, use_color=use_color)
            )
            print(f"  [{mark}] {r.stage:<3} {r.adapter:<16} {r.detail}")
        print()

    if report.personas:
        print(f"Personas ({len(report.personas)}): {', '.join(report.personas)}")
        print()

    if report.voices:
        print(f"Voices ({len(report.voices)}): {', '.join(report.voices)}")
        print()

    if report.clones:
        assigned = report.clone_assignments
        # Show each clone, noting any persona it's assigned to (M5).
        by_clone = {name: pid for pid, name in assigned.items()}
        listed = ", ".join(
            f"{name} -> {by_clone[name]}" if name in by_clone else name for name in report.clones
        )
        print(f"Clones ({len(report.clones)}): {listed}")
        print()

    if report.finetuned:
        by_voice = {name: pid for pid, name in report.finetuned_assignments.items()}
        listed = ", ".join(
            f"{name} -> {by_voice[name]}" if name in by_voice else name
            for name in report.finetuned
        )
        print(f"Fine-tuned voices ({len(report.finetuned)}): {listed}")
        print()

    if report.memory_dir:
        enc = "encrypted" if report.memory_encrypted else "plaintext"
        print(f"Memory: {report.memory_dir}  ({enc}, {report.memory_users} user(s))")
        print()

    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  {_c('!', _YELLOW, use_color=use_color)} {w}")
        print()

    status = (
        _c("PASS", _GREEN, use_color=use_color)
        if report.ok
        else _c("FAIL", _RED, use_color=use_color)
    )
    print(f"Result: {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="personavoice.server")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate config and load (stub) adapters, then exit",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run the live LiveKit streaming agent (M3); needs the `livekit` extra",
    )
    parser.add_argument(
        "--token-server",
        action="store_true",
        help="run the HTTP token server that mints LiveKit join tokens for clients (M6)",
    )
    parser.add_argument(
        "--backend",
        choices=("mac", "cuda"),
        default=None,
        help="override the BACKEND env var",
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = parser.parse_args(argv)

    use_color = sys.stdout.isatty() and not args.no_color

    from ..obs import configure_logging

    configure_logging()

    try:
        settings = Settings.load(backend=args.backend)
    except ConfigError as exc:
        print(_c("ERROR", _RED, use_color=use_color), exc)
        return 2

    if args.check:
        report = run_check(settings)
        _print_report(report, settings, use_color=use_color)
        return 0 if report.ok else 1

    if args.token_server:
        # Pure stdlib HTTP server — no heavy extra needed; mints LiveKit join tokens.
        from . import token_server

        print(f"Starting token server (backend={settings.backend})...")
        token_server.run(settings)
        return 0

    if args.serve:
        # The LiveKit worker takes over argv/lifecycle, so import lazily and hand off.
        from ..orchestrator import agent

        print(f"Starting LiveKit streaming agent (backend={settings.backend})...")
        try:
            agent.run()
        except RuntimeError as exc:  # e.g. the `livekit` extra isn't installed
            print(_c("ERROR", _RED, use_color=use_color), exc)
            return 2
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
