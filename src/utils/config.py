from pathlib import Path


def load_config(path):
    if path is None:
        return {}
    return _parse_simple_yaml(Path(path).read_text())


def get_config_value(config, dotted_path, default=None):
    current = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def expand_path(value):
    if value is None:
        return None
    return str(Path(value).expanduser())


def _parse_simple_yaml(text):
    root = {}
    stack = [(0, root)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if ":" not in stripped:
            raise ValueError(f"Invalid config line: {raw_line!r}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if not value:
            new_mapping = {}
            current[key] = new_mapping
            stack.append((indent + 2, new_mapping))
            continue

        current[key] = _parse_scalar(value)

    return root


def _parse_scalar(value):
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value
