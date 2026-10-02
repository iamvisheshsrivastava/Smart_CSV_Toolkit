"""Best-effort hardening for executing LLM-generated Python (issue #4).

This is defence in depth, not a true sandbox: code is statically validated
(AST) and run with a curated ``__builtins__``. For strong isolation run the
app in a container without secrets/network egress.
"""
import ast
import builtins as _b

ALLOWED_IMPORTS = {
    "pandas", "numpy", "re", "math", "datetime", "string", "collections",
    "statistics", "sklearn", "imblearn", "nltk", "matplotlib", "seaborn",
    "dateparser", "tldextract", "geopy",
}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "help",
    "os", "sys", "subprocess", "socket", "shutil", "importlib", "builtins",
}
FORBIDDEN_ATTRS = {
    "system", "popen", "eval", "query", "read_pickle", "to_pickle", "to_csv", "to_excel",
    "to_json", "to_parquet", "to_sql", "to_feather", "to_hdf", "to_html",
    "to_clipboard", "read_clipboard", "savefig", "download", "getenv", "environ",
    "load_model", "from_pretrained", "format", "format_map",
}
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
    "isinstance", "issubclass", "len", "list", "map", "max", "min", "range",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "print", "bytes", "chr", "ord", "divmod", "pow", "repr", "frozenset", "type",
    "hasattr", "callable", "iter", "next", "format", "hash", "id",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "AttributeError", "RuntimeError", "StopIteration",
    "True", "False", "None",
]


def validate_code(code: str) -> None:
    """Raise ValueError if the code uses constructs we refuse to execute."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Generated code has a syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("Relative imports are not allowed in generated code.")
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import of '{m}' is not allowed in generated code.")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or (node.id.startswith("__") and node.id.endswith("__")):
                raise ValueError(f"Use of '{node.id}' is not allowed in generated code.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS or node.attr.startswith("read_"):
                raise ValueError(f"Access to attribute '{node.attr}' is not allowed in generated code.")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal is not allowed in generated code.")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed.")
    return _b.__import__(name, globals, locals, fromlist, level)


def safe_builtins() -> dict:
    d = {n: getattr(_b, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_b, n)}
    d["__import__"] = _safe_import
    return d


def safe_exec(code: str, global_vars: dict, local_vars: dict) -> None:
    """Validate then exec ``code`` with restricted builtins."""
    validate_code(code)
    global_vars["__builtins__"] = safe_builtins()
    exec(code, global_vars, local_vars)
