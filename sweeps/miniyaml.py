"""Minimal YAML-subset reader/writer used when PyYAML is unavailable.

The sweep platform/spec files deliberately restrict themselves to this subset
so the runner stays stdlib-only on any platform:
  - comments (# ...), blank lines
  - key: scalar        (str, int, float, bool, null; optional quotes)
  - key: [a, b, c]     (inline list of scalars)
  - key: {}            (empty dict)
  - block lists of scalars ("- item" indented under "key:")
  - one level of nested dict (indented "sub: value" under "key:")
Anything else raises, loudly — extend the subset before extending the files.
"""


def _scalar(tok):
    tok = tok.strip()
    if tok.startswith(("'", '"')) and tok.endswith(tok[0]) and len(tok) >= 2:
        return tok[1:-1]
    low = tok.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~", ""):
        return None
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def _strip_comment(line):
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def loads(text):
    root = {}
    stack = [(0, root)]  # (indent, container)
    pending_key = None  # key awaiting a block value
    lines = [_strip_comment(ln) for ln in text.splitlines()]
    for ln in lines:
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip())
        body = ln.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        container = stack[-1][1]
        if body.startswith("- "):
            if not isinstance(container, list):
                if pending_key is None:
                    raise ValueError(f"list item outside a list: {ln!r}")
                new = []
                parent = stack[-1][1]
                parent[pending_key] = new
                stack.append((indent, new))
                pending_key = None
                container = new
            container.append(_scalar(body[2:]))
            continue
        if ":" not in body:
            raise ValueError(f"unparseable line: {ln!r}")
        key, _, val = body.partition(":")
        key, val = key.strip(), val.strip()
        if isinstance(container, list):
            stack.pop()
            container = stack[-1][1]
        if pending_key is not None and indent > stack[-1][0]:
            new = {}
            container[pending_key] = new
            stack.append((indent, new))
            container = new
            pending_key = None
        elif pending_key is not None:
            container[pending_key] = None  # bare "key:" with no block
            pending_key = None
        if not val:
            pending_key = key
            continue
        if val == "{}":
            container[key] = {}
        elif val == "[]":
            container[key] = []
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            container[key] = [_scalar(x) for x in inner.split(",")] if inner else []
        elif val.startswith("{"):
            raise ValueError(f"inline dicts beyond {{}} unsupported: {ln!r}")
        else:
            container[key] = _scalar(val)
    if pending_key is not None:
        stack[-1][1] if stack else root
        # trailing bare "key:" -> null
        (stack[-1][1] if stack else root)[pending_key] = None
    return root


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, str):
        needs_quote = (
            v == ""
            or v != v.strip()
            or any(c in v for c in ":#{}[]'\",")
            or _scalar(v) != v  # would re-parse as int/float/bool/null ("001", "yes")
        )
        return f'"{v}"' if needs_quote else v
    return str(v)


def dumps(obj, indent=0):
    lines = []
    pad = " " * indent
    for k in sorted(obj):
        v = obj[k]
        if isinstance(v, dict):
            if not v:
                lines.append(f"{pad}{k}: {{}}")
            else:
                lines.append(f"{pad}{k}:")
                lines.append(dumps(v, indent + 2))
        elif isinstance(v, (list, tuple)):
            if not v:
                lines.append(f"{pad}{k}: []")
            else:
                lines.append(f"{pad}{k}:")
                for item in v:
                    lines.append(f"{pad}  - {_fmt(item)}")
        else:
            lines.append(f"{pad}{k}: {_fmt(v)}")
    return "\n".join(lines)
