from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_file(path: str | Path, *, label: str) -> Any:
    file_path = Path(path)
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {file_path}") from exc


def read_optional_json_input(*, inline_json: str | None, file_path: str | None, label: str) -> dict[str, Any]:
    if inline_json is not None:
        try:
            data = json.loads(inline_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not valid JSON") from exc
        if not isinstance(data, dict):
            raise TypeError(f"{label} must be a JSON object")
        return data

    if file_path is None:
        return {}

    data = read_json_file(file_path, label=label)
    if not isinstance(data, dict):
        raise TypeError(f"{label} must be a JSON object")
    return data


def read_messages_file(path: str | Path) -> list[dict[str, Any]]:
    data = read_json_file(path, label="messages-file")
    if not isinstance(data, list):
        raise TypeError("messages-file must be a JSON array")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"messages-file[{i}] must be a JSON object")
    return data


def read_secret_bits(*, secret_bits: str | None, secret_bits_file: str | None) -> str:
    if secret_bits is not None:
        bits = secret_bits.strip()
    elif secret_bits_file is not None:
        try:
            bits = Path(secret_bits_file).read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"secret-bits-file not found: {secret_bits_file}") from exc
    else:
        raise ValueError("one of --secret-bits or --secret-bits-file is required")

    if set(bits) - {"0", "1"}:
        raise ValueError("secret bits must contain only '0' and '1'")
    return bits


def read_generated_token_ids_file(path: str | Path) -> list[int]:
    data = read_json_file(path, label="generated-token-ids-file")
    if isinstance(data, dict):
        if "generated_token_ids" not in data:
            raise ValueError("generated-token-ids-file JSON object must contain 'generated_token_ids'")
        data = data["generated_token_ids"]
    if not isinstance(data, list):
        raise TypeError("generated-token-ids-file must be a JSON array or an encode result object")

    token_ids: list[int] = []
    for i, value in enumerate(data):
        if not isinstance(value, int):
            raise TypeError(f"generated-token-ids-file[{i}] must be an integer")
        token_ids.append(value)
    return token_ids


def write_json_output(payload: Any, *, output_file: str | None, quiet: bool) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_file is not None:
        Path(output_file).write_text(rendered + "\n", encoding="utf-8")
    if not quiet:
        print(rendered)
