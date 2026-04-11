from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from .commands import run_algorithms, run_decode, run_encode


def _add_shared_runtime_args(parser: argparse.ArgumentParser, *, include_max_new_tokens: bool) -> None:
    parser.add_argument("--algorithm", required=True, help="Algorithm name, for example ac or adg")
    parser.add_argument("--model", required=True, help="Hugging Face model name or local model path")
    parser.add_argument("--messages-file", required=True, help="Path to a JSON messages file")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--precision", type=int, default=52)
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config-json", help="Inline JSON object for algorithm config")
    config_group.add_argument("--config-file", help="Path to a JSON file for algorithm config")
    material_group = parser.add_mutually_exclusive_group()
    material_group.add_argument("--material-json", help="Inline JSON object for material")
    material_group.add_argument("--material-file", help="Path to a JSON file for material")
    parser.add_argument("--output-file", help="Optional path to write JSON output")
    parser.add_argument("--quiet", action="store_true", help="Suppress JSON stdout output")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model/tokenizer loaders",
    )
    if include_max_new_tokens:
        parser.add_argument("--max-new-tokens", type=int, default=128)
        parser.add_argument(
            "--stop-on-eos",
            action="store_true",
            help="Stop generation when eos token is sampled",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StegoKit command line interface")
    subparsers = parser.add_subparsers(dest="command")

    algorithms_parser = subparsers.add_parser("algorithms", help="List built-in algorithms")
    algorithms_parser.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON instead of a table")
    algorithms_parser.add_argument("--output-file", help="Optional path to write JSON output")
    algorithms_parser.add_argument("--quiet", action="store_true", help="Suppress stdout output")
    algorithms_parser.set_defaults(func=run_algorithms)

    encode_parser = subparsers.add_parser("encode", help="Embed secret bits into generated text")
    _add_shared_runtime_args(encode_parser, include_max_new_tokens=True)
    secret_group = encode_parser.add_mutually_exclusive_group(required=True)
    secret_group.add_argument("--secret-bits", help="Bit string containing only 0 and 1")
    secret_group.add_argument("--secret-bits-file", help="Path to a text file containing the secret bit string")
    encode_parser.set_defaults(func=run_encode)

    decode_parser = subparsers.add_parser("decode", help="Recover secret bits from generated token ids")
    _add_shared_runtime_args(decode_parser, include_max_new_tokens=False)
    decode_parser.add_argument(
        "--generated-token-ids-file",
        required=True,
        help="Path to a JSON file containing generated token ids or an encode result JSON",
    )
    decode_parser.add_argument("--max-bits", type=int, default=None)
    decode_parser.set_defaults(func=run_decode)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    logging.basicConfig(level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO)

    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
