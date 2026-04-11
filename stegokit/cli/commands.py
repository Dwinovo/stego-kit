from __future__ import annotations

from dataclasses import asdict
from typing import Any

from stegokit.core.stego_dispatcher import StegoDispatcher
from stegokit.core.stego_registry import StegoAlgorithmRegistry

from . import builders, io


def run_algorithms(args) -> int:
    registry = StegoAlgorithmRegistry.default()
    rows = []
    for name, spec in sorted(registry.specs().items()):
        rows.append(
            {
                "name": name,
                "paradigm": spec.paradigm.value,
                "encode_config_type": spec.encode_config_type.__name__,
                "decode_config_type": spec.decode_config_type.__name__,
                "encode_material_type": spec.encode_material_type.__name__,
                "decode_material_type": spec.decode_material_type.__name__,
            }
        )

    if getattr(args, "json_output", False):
        io.write_json_output(rows, output_file=args.output_file, quiet=args.quiet)
        return 0

    if args.output_file is not None:
        io.write_json_output(rows, output_file=args.output_file, quiet=True)

    if args.quiet:
        return 0

    headers = [
        "name",
        "paradigm",
        "encode_config",
        "decode_config",
        "encode_material",
        "decode_material",
    ]
    table_rows = [
        [
            row["name"],
            row["paradigm"],
            row["encode_config_type"],
            row["decode_config_type"],
            row["encode_material_type"],
            row["decode_material_type"],
        ]
        for row in rows
    ]
    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def render(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(render(headers))
    print(render(["-" * width for width in widths]))
    for row in table_rows:
        print(render(row))
    return 0


def run_encode(args) -> int:
    registry = StegoAlgorithmRegistry.default()
    spec = registry.get_spec(args.algorithm)
    messages = io.read_messages_file(args.messages_file)
    secret_bits = io.read_secret_bits(secret_bits=args.secret_bits, secret_bits_file=args.secret_bits_file)
    config_payload = io.read_optional_json_input(
        inline_json=args.config_json,
        file_path=args.config_file,
        label="config",
    )
    material_payload = io.read_optional_json_input(
        inline_json=args.material_json,
        file_path=args.material_file,
        label="material",
    )

    generation = builders.build_generation_config(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        precision=args.precision,
        stop_on_eos=args.stop_on_eos,
        max_new_tokens=args.max_new_tokens,
    )
    runtime = builders.build_runtime_context(
        model_name_or_path=args.model,
        messages=messages,
        generation=generation,
        trust_remote_code=args.trust_remote_code,
    )
    config = builders.build_config(spec.encode_config_type, config_payload)
    material = builders.build_material(spec.encode_material_type, material_payload)

    dispatcher = StegoDispatcher(verbose=not args.quiet)
    result = dispatcher.embed(
        algorithm=args.algorithm,
        runtime=runtime,
        secret_bits=secret_bits,
        config=config,
        material=material,
    )
    payload: dict[str, Any] = {
        "algorithm": spec.name,
        **asdict(result),
    }
    io.write_json_output(payload, output_file=args.output_file, quiet=args.quiet)
    return 0


def run_decode(args) -> int:
    registry = StegoAlgorithmRegistry.default()
    spec = registry.get_spec(args.algorithm)
    messages = io.read_messages_file(args.messages_file)
    generated_token_ids = io.read_generated_token_ids_file(args.generated_token_ids_file)
    config_payload = io.read_optional_json_input(
        inline_json=args.config_json,
        file_path=args.config_file,
        label="config",
    )
    material_payload = io.read_optional_json_input(
        inline_json=args.material_json,
        file_path=args.material_file,
        label="material",
    )

    generation = builders.build_generation_config(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        precision=args.precision,
        stop_on_eos=None,
        max_new_tokens=128,
    )
    runtime = builders.build_runtime_context(
        model_name_or_path=args.model,
        messages=messages,
        generation=generation,
        trust_remote_code=args.trust_remote_code,
    )
    config = builders.build_config(spec.decode_config_type, config_payload)
    material = builders.build_material(spec.decode_material_type, material_payload)

    dispatcher = StegoDispatcher(verbose=not args.quiet)
    result = dispatcher.extract(
        algorithm=args.algorithm,
        runtime=runtime,
        generated_token_ids=generated_token_ids,
        max_bits=args.max_bits,
        config=config,
        material=material,
    )
    payload: dict[str, Any] = {
        "algorithm": spec.name,
        **asdict(result),
    }
    io.write_json_output(payload, output_file=args.output_file, quiet=args.quiet)
    return 0
