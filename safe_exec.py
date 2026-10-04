"""Hardening for executing LLM-generated Python (issue #4).

Two layers are combined:

1. Static AST validation + a curated ``__builtins__`` (``validate_code`` /
   ``safe_builtins`` / ``safe_exec``) — blocks disallowed imports, dunder
   access, and known-dangerous attributes/names.
2. Process-level isolation (``run_sandboxed``) — the validated code is
   executed in a *separate, short-lived subprocess* (not the Streamlit
   server process), with:
     - a wall-clock timeout (the subprocess is killed if it runs too long),
     - a CPU-time rlimit and an address-space (memory) rlimit on POSIX
       (Linux/macOS — e.g. the Render deployment), via ``resource.setrlimit``,
     - no inherited Streamlit/session state, secrets, or open file handles
       beyond what is explicitly passed in (the child only receives the
       code string and the DataFrame to operate on; it rebuilds its own
       module context from scratch),
     - no filesystem or network access beyond what layer 1 already blocks
       (``os``, ``subprocess``, ``socket``, ``requests`` etc. are not in the
       import allowlist, so the child process has no way to reach the
       filesystem or network even though it runs in the same container).

This is still not a full gVisor/Docker/nsjail-style sandbox: the subprocess
shares the same container, filesystem, and network namespace as the main
app (it just doesn't have Python-level handles to use them, and the AST
allowlist stops it from importing anything that would). A compromise of the
allowlist (e.g. a future "allowed" library that itself shells out) would
still reach the host. True OS-level isolation (a gVisor/Docker/nsjail
sandbox with its own network namespace and read-only filesystem) is the
right long-term fix but isn't available on every deployment target (e.g.
Render's free tier cannot spawn nested containers) — see issue #4.
"""
import ast
import builtins as _b
import multiprocessing as mp
import io

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


# ---------------------------------------------------------------------------
# Process-level isolation (defence in depth on top of safe_exec above).
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_CPU_SECONDS = 15
DEFAULT_MEMORY_MB = 768


def _build_sandbox_context(kind: str) -> dict:
    """Rebuild the module/class context for ``kind`` *inside* the child
    process, rather than pickling module objects from the parent."""
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    ctx = {
        "pd": pd,
        "np": np,
        "SimpleImputer": SimpleImputer,
        "StandardScaler": StandardScaler,
        "MinMaxScaler": MinMaxScaler,
        "Binarizer": Binarizer,
        "SMOTE": SMOTE,
        "RandomOverSampler": RandomOverSampler,
        "RandomUnderSampler": RandomUnderSampler,
    }

    if kind in ("custom_cleaning", "plot"):
        import re
        from datetime import datetime
        ctx.update({"re": re, "datetime": datetime})

    if kind == "custom_cleaning":
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import geopy
        from geopy.distance import geodesic
        import tldextract

        try:
            stops = set(stopwords.words("english"))
        except LookupError:
            stops = set()

        ctx.update({
            "nltk": nltk,
            "geopy": geopy,
            "geodesic": geodesic,
            "stops": stops,
            "word_tokenize": word_tokenize,
            "tldextract": tldextract,
        })
        try:
            ctx["transformers"] = __import__("transformers")
        except ImportError:
            pass

    if kind == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        ctx.update({"plt": plt, "sns": sns})

    return ctx


def _apply_resource_limits(memory_mb: int, cpu_seconds: int) -> None:
    """Best-effort CPU/memory caps. Only available on POSIX (Linux/macOS —
    e.g. the Render deployment). No-op on Windows, where ``resource`` does
    not exist; the wall-clock timeout in ``run_sandboxed`` still applies
    there."""
    try:
        import resource
        if memory_mb:
            limit_bytes = memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except (ValueError, OSError):
                pass
        if cpu_seconds:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            except (ValueError, OSError):
                pass
    except ImportError:
        pass


def _sandbox_worker(kind, code, df, memory_mb, cpu_seconds, conn):
    try:
        _apply_resource_limits(memory_mb, cpu_seconds)
        validate_code(code)
        global_vars = _build_sandbox_context(kind)
        global_vars["__builtins__"] = safe_builtins()
        local_vars = {"df": df}
        exec(code, global_vars, local_vars)

        if kind == "plot":
            plt = global_vars["plt"]
            fig = plt.gcf()
            fig.set_size_inches(6, 4)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120)
            plt.close(fig)
            conn.send(("ok", buf.getvalue()))
        else:
            conn.send(("ok", local_vars.get("df")))
    except MemoryError:
        conn.send(("error", "Code exceeded the memory limit and was terminated."))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_sandboxed(
    kind: str,
    code: str,
    df,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
):
    """Run LLM-generated ``code`` against ``df`` in an isolated subprocess.

    ``kind`` selects which module context to rebuild in the child
    (``"pipeline"``, ``"custom_cleaning"``, or ``"plot"``). For ``"plot"``,
    returns PNG bytes; otherwise returns the resulting DataFrame.

    Raises RuntimeError on any failure, including timeout, validation
    failure, or an exception raised by the generated code itself.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_sandbox_worker,
        args=(kind, code, df, memory_mb, cpu_seconds, child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()

    status, payload = None, None
    try:
        if parent_conn.poll(timeout):
            try:
                status, payload = parent_conn.recv()
            except EOFError:
                status, payload = "error", "The sandboxed process crashed without returning a result."
        else:
            status, payload = "error", f"Execution timed out after {timeout}s and was terminated."
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.terminate()
            proc.join(3)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
        else:
            proc.join(2)

    if status != "ok":
        raise RuntimeError(payload)
    return payload
