"""Command line entry point.

  python -m rts serve                       watch two scripted bots
  python -m rts serve --red llm --blue boom  Claude vs a heuristic bot
  python -m rts headless --matches 20        run a batch and print win rates
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from . import config as cfg
from .commanders import DEFAULT_MODEL, PERSONALITIES
from .match import run_headless
from .server import MatchRunner, serve

SPEC_HELP = ("one of " + ", ".join(PERSONALITIES) +
             ", or 'llm' / 'llm:<persona description>'")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--red", default="balanced", help=SPEC_HELP)
    parser.add_argument("--blue", default="rush", help=SPEC_HELP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--think-interval", type=int, default=cfg.THINK_INTERVAL_TICKS,
                        help="ticks between commander turns (10 ticks = 1 second)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rts", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("serve", help="run the spectator server")
    _add_common(play)
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=8765)
    play.add_argument("--speed", type=float, default=1.0)
    play.add_argument("--model", default=DEFAULT_MODEL, help="model for llm commanders")
    play.add_argument("--red-persona", default="", help="character notes for the red agent")
    play.add_argument("--blue-persona", default="", help="character notes for the blue agent")
    play.add_argument("--max-calls", type=int, default=None,
                      help="cap API calls per agent per match, then fall back to scripted")
    play.add_argument("--record", default="", metavar="DIR",
                      help="write every match to DIR as JSONL snapshots")
    play.add_argument("--no-autorestart", action="store_true",
                      help="stop after one match instead of starting a new one")

    batch = sub.add_parser("headless", help="run matches with no server or UI")
    _add_common(batch)
    batch.add_argument("--matches", type=int, default=10)
    batch.add_argument("--json", action="store_true", help="emit one JSON record per match")

    args = parser.parse_args(argv)

    if args.command == "serve":
        runner = MatchRunner(
            args.red, args.blue, seed=args.seed, speed=args.speed,
            think_interval=args.think_interval, model=args.model,
            personas=(args.red_persona, args.blue_persona),
            max_calls=args.max_calls, autorestart=not args.no_autorestart,
            record_dir=args.record)
        serve(runner, args.host, args.port)
        return 0

    wins: collections.Counter[str] = collections.Counter()
    for i in range(args.matches):
        result = run_headless(args.red, args.blue, seed=args.seed + i,
                              quiet=args.json, think_interval=args.think_interval)
        if args.json:
            print(json.dumps(result))
        wins[result["winner"] or "draw"] += 1
    total = max(args.matches, 1)
    print(f"\n{args.red} (RED) {wins['RED']}/{total}   "
          f"{args.blue} (BLUE) {wins['BLUE']}/{total}   draws {wins['draw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
