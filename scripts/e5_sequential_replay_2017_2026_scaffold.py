from __future__ import annotations

import argparse


READY_MESSAGE = "E5_REPLAY_SCAFFOLD_READY_NOT_EXECUTED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planning scaffold for E5 sequential replay. Does not run prediction replay yet.",
    )
    parser.add_argument("--start-draw", type=int, default=None)
    parser.add_argument("--end-draw", type=int, default=None)
    parser.add_argument("--memory-path", default="artifacts/e5/segment_memory_registry.json")
    parser.add_argument("--no-write", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="Reserved for future implementation.")
    return parser.parse_args()


def intended_loop() -> list[str]:
    return [
        "predict draw N using knowledge <= N-1",
        "verify draw N after completion",
        "run E5 segment attribution",
        "update observation-only segment memory",
        "advance to next draw",
    ]


def main() -> int:
    args = parse_args()
    print(READY_MESSAGE)
    print(f"start_draw={args.start_draw}")
    print(f"end_draw={args.end_draw}")
    print(f"memory_path={args.memory_path}")
    print(f"no_write={args.no_write}")
    print("intended_loop:")
    for step in intended_loop():
        print(f"- {step}")
    if args.execute:
        raise SystemExit("Full E5 replay is not implemented in this scaffold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
