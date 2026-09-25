"""AST-based introspection of user Python files into CapabilityDrafts.

Deliberately **never imports or executes** the file being scanned: a user's
module may have heavy or missing dependencies, and onboarding should be safe to
run against arbitrary code. Everything is inferred from the parse tree:

* argument names, annotations and defaults  -> tool inputs
* return annotation + docstring 'Returns:'  -> tool outputs
* docstring summary                         -> tool description
* function/module name heuristics           -> suggested subagent + category

Every inference the scanner is unsure about is recorded in ``draft.notes`` so the
agent can surface it for confirmation instead of silently guessing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

from common.tool_categories import BUILTIN_CATEGORY_NAMES, category_specs
from onboarding.models import (
    CapabilityDraft,
    OutputDraft,
    OutputType,
    ParamDraft,
    ParamType,
    ScanResult,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directory under tools/ that each subagent's capabilities are filed in.
SUBAGENT_TOOL_DIRECTORY = {
    "data_discovery": "loaders",
    "extraction": "extractors",
    "diagnostics": "analyzers",
    "water_cycle": "analyzers",
    "visualization": "plotters",
}

ANNOTATION_TO_PARAM_TYPE = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "Path": "path",
    "list": "list[str]",
    "List": "list[str]",
    "list[str]": "list[str]",
    "List[str]": "list[str]",
    "list[int]": "list[int]",
    "List[int]": "list[int]",
}

# Argument names that almost always mean "a file on disk".
PATH_NAME_PATTERN = re.compile(r"(^|_)(file|path|dir|filename|filepath)s?$")

RETURN_ANNOTATION_TO_OUTPUT = {
    "DataFrame": ("table_file", "csv"),
    "Series": ("table_file", "csv"),
    "Dataset": ("output_file", "netcdf"),
    "DataArray": ("output_file", "netcdf"),
    "Figure": ("plot_file", "png"),
}

VISUALIZATION_HINTS = ("plot", "figure", "chart", "map", "heatmap", "render", "draw")
EXTRACTION_HINTS = ("extract", "subset", "regrid", "match", "slice", "sample")
DISCOVERY_HINTS = ("fetch", "download", "catalog", "search", "discover", "load_meta")
WATER_CYCLE_HINTS = ("basin", "budget", "water_balance", "runoff_ratio", "discharge")


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------


def _docstring_summary(docstring: Optional[str]) -> str:
    if not docstring:
        return ""
    parts: List[str] = []
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            break
        parts.append(stripped)
    return " ".join(parts)


def _docstring_sections(docstring: Optional[str]) -> Dict[str, List[str]]:
    """Split a Google-style docstring into ``{section_lower: [lines]}``."""

    if not docstring:
        return {}
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw in docstring.splitlines():
        header = re.match(r"^\s*(Args|Arguments|Returns|Yields|Raises)\s*:\s*$", raw)
        if header:
            current = str(header.group(1)).lower()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(raw)
    return sections


def _arg_descriptions(docstring: Optional[str]) -> Dict[str, str]:
    """Map argument name -> description from the Args: section."""

    sections = _docstring_sections(docstring)
    lines = sections.get("args") or sections.get("arguments") or []
    descriptions: Dict[str, str] = {}
    current: Optional[str] = None
    for raw in lines:
        match = re.match(r"^\s{2,}(\*{0,2}\w+)\s*(\([^)]*\))?\s*:\s*(.*)$", raw)
        if match:
            current = str(match.group(1)).lstrip("*")
            descriptions[current] = str(match.group(3)).strip()
        elif current and raw.strip():
            descriptions[current] = f"{descriptions[current]} {raw.strip()}".strip()
    return descriptions


def _return_description(docstring: Optional[str]) -> str:
    lines = _docstring_sections(docstring).get("returns") or []
    return " ".join(line.strip() for line in lines if line.strip())


# ---------------------------------------------------------------------------
# Annotation / default handling
# ---------------------------------------------------------------------------


def _annotation_text(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return ""


def _strip_optional(annotation: str) -> Tuple[str, bool]:
    """Return ``(inner_type, was_optional)`` for ``Optional[X]`` / ``X | None``."""

    annotation = annotation.strip()
    optional = re.match(r"^Optional\[(.+)\]$", annotation)
    if optional:
        return str(optional.group(1)).strip(), True
    if "|" in annotation:
        parts = [p.strip() for p in annotation.split("|")]
        non_none = [p for p in parts if p != "None"]
        if len(non_none) == 1:
            return non_none[0], len(non_none) != len(parts)
    return annotation, False


def _literal_default(node: Optional[ast.AST]) -> Tuple[Any, bool]:
    """Return ``(value, is_literal)`` for a default expression."""

    if node is None:
        return None, False
    try:
        return ast.literal_eval(node), True
    except Exception:
        return None, False


def _param_type_for(name: str, annotation: str, default: Any) -> Tuple[str, List[str]]:
    notes: List[str] = []
    base, _ = _strip_optional(annotation) if annotation else ("", False)
    base = base if isinstance(base, str) else ""
    base = base.replace(" ", "")

    if base in ANNOTATION_TO_PARAM_TYPE:
        param_type = ANNOTATION_TO_PARAM_TYPE[base]
    elif isinstance(default, bool):
        param_type = "bool"
    elif isinstance(default, int):
        param_type = "int"
    elif isinstance(default, float):
        param_type = "float"
    elif isinstance(default, (list, tuple)):
        param_type = "list[str]"
    else:
        param_type = "str"
        if base:
            notes.append(
                f"input '{name}': annotation '{annotation}' not recognized, assumed str"
            )
        else:
            notes.append(f"input '{name}': no annotation, assumed str")

    # File-ish names are worth promoting to 'path' so workflow references and
    # existence checks behave like every other ESMFlow tool.
    if param_type == "str" and PATH_NAME_PATTERN.search(name):
        param_type = "path"

    return param_type, notes


# ---------------------------------------------------------------------------
# Output inference
# ---------------------------------------------------------------------------


def _outputs_from_returned_dict(
    func: ast.FunctionDef,
) -> List[Tuple[str, Optional[ast.AST]]]:
    """Collect ``{'key': value}`` pairs from the function's return statements."""

    pairs: List[Tuple[str, Optional[ast.AST]]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                pairs.append((key.value, value))
        if pairs:
            break
    return pairs


#: Python literal types that map directly onto scalar output types.
_CONSTANT_OUTPUT_TYPES: Dict[type, str] = {bool: "int", int: "int", float: "float", str: "str"}


def _value_output_type(value: Optional[ast.AST]) -> str:
    """Guess an output type from the expression assigned to a dict key.

    Literal constants (``3``, ``-1.5``, ``"ok"``) and explicit coercions
    (``int(...)``, ``float(...)``, ``str(...)``) are strong evidence. Anything
    else returns ``""`` so the caller can fall back to its own heuristics.
    """

    if value is None:
        return "str"

    # ``-1.5`` parses as UnaryOp(USub, Constant(1.5)); look through the sign.
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
        value = value.operand
    if isinstance(value, ast.Constant):
        # bool is a subclass of int, so check it via the exact type.
        return _CONSTANT_OUTPUT_TYPES.get(type(value.value), "")

    text = _annotation_text(value)
    if re.match(r"^int\(", text):
        return "int"
    if re.match(r"^float\(", text):
        return "float"
    if re.match(r"^str\(", text):
        return "str"
    return ""


def _infer_outputs(
    func: ast.FunctionDef, docstring: Optional[str]
) -> Tuple[List[OutputDraft], List[str]]:
    notes: List[str] = []
    return_annotation, _ = _strip_optional(_annotation_text(func.returns))
    return_annotation = return_annotation if isinstance(return_annotation, str) else ""
    base = return_annotation.split("[")[0].split(".")[-1]
    return_description = _return_description(docstring)

    dict_pairs = _outputs_from_returned_dict(func)
    if dict_pairs:
        outputs: List[OutputDraft] = []
        for key, value in dict_pairs:
            guessed = _value_output_type(value)
            if guessed:
                out_type = guessed
            elif key.endswith("_file") or key.endswith("_path"):
                out_type = "csv"
                notes.append(
                    f"output '{key}': assumed csv from its name; change to "
                    "netcdf/png if the function produces those"
                )
            else:
                out_type = "float"
                notes.append(f"output '{key}': type guessed as float")
            outputs.append(
                OutputDraft(
                    name=key, type=cast(OutputType, out_type), description=""
                )
            )
        return outputs, notes

    if base in RETURN_ANNOTATION_TO_OUTPUT:
        name, out_type = RETURN_ANNOTATION_TO_OUTPUT[base]
        return [
            OutputDraft(
                name=name,
                type=cast(OutputType, out_type),
                description=return_description,
            )
        ], notes

    if base == "dict":
        notes.append(
            "function returns a dict but its keys could not be read statically; "
            "declare the output names explicitly before registering"
        )
        return [], notes

    notes.append(
        "could not infer outputs from the signature; declare at least one output "
        "(name + type) before registering"
    )
    return [], notes


# ---------------------------------------------------------------------------
# Routing heuristics
# ---------------------------------------------------------------------------


def _suggest_subagent(
    func_name: str, module_name: str, outputs: List[OutputDraft]
) -> Tuple[str, List[str]]:
    haystack = f"{func_name} {module_name}".lower()
    notes: List[str] = []

    if any(out.type == "png" for out in outputs) or any(
        hint in haystack for hint in VISUALIZATION_HINTS
    ):
        return "visualization", notes
    if any(hint in haystack for hint in DISCOVERY_HINTS):
        return "data_discovery", notes
    if any(hint in haystack for hint in EXTRACTION_HINTS):
        return "extraction", notes
    if any(hint in haystack for hint in WATER_CYCLE_HINTS):
        return "water_cycle", notes

    notes.append(
        "subagent defaulted to 'diagnostics'; reassign if this capability belongs "
        f"to another category ({', '.join(BUILTIN_CATEGORY_NAMES)}) or a new one"
    )
    return "diagnostics", notes


def category_for_subagent(subagent: str) -> str:
    """Which ``tools/<dir>/`` an adapter for ``subagent`` should be written to."""

    return SUBAGENT_TOOL_DIRECTORY.get(subagent, "analyzers")


def _tool_name_for(func_name: str) -> str:
    return func_name.lstrip("_")


def _module_path_for(source: Path) -> str:
    """Dotted import path for a file inside the repo."""

    resolved = source.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return resolved.stem
    parts = list(relative.with_suffix("").parts)
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_source(
    source: str | Path, function_name: Optional[str] = None
) -> ScanResult:
    """Parse a Python file and draft a capability per public function.

    Args:
        source: Path to the user's ``.py`` file (inside the repo).
        function_name: Only draft this function when given.

    Returns:
        A :class:`ScanResult` with one draft per eligible function. Nothing is
        written to disk and the module is never imported.
    """

    raw_source = str(source)
    path = Path(raw_source)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists() and "." in raw_source and not raw_source.endswith(".py"):
        path = REPO_ROOT.joinpath(*raw_source.split(".")).with_suffix(".py")
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such source file: {source}")
    if path.suffix != ".py":
        raise ValueError(f"Only .py files can be scanned, got: {path.name}")

    tree = ast.parse(path.read_text(), filename=str(path))
    module_path = _module_path_for(path)
    module_docstring = ast.get_docstring(tree) or ""
    known_categories = set(category_specs())

    result = ScanResult(source=str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path))

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.AsyncFunctionDef):
            result.skipped.append(f"{node.name} (async functions are not supported)")
            continue
        if function_name and node.name != function_name:
            continue
        if not function_name and node.name.startswith("_"):
            result.skipped.append(f"{node.name} (private helper)")
            continue

        draft, notes = _draft_for_function(node, module_path, module_docstring)
        if draft.subagent not in known_categories:  # pragma: no cover - defensive
            notes.append(
                f"suggested subagent '{draft.subagent}' is not registered yet"
            )
        draft.notes = notes
        result.drafts.append(draft)

    if function_name and not result.drafts:
        raise ValueError(f"Function '{function_name}' not found in {path.name}")

    return result


def _draft_for_function(
    func: ast.FunctionDef, module_path: str, module_docstring: str
) -> Tuple[CapabilityDraft, List[str]]:
    docstring = ast.get_docstring(func)
    notes: List[str] = []

    description = _docstring_summary(docstring)
    if not description:
        description = _docstring_summary(module_docstring)
        notes.append(
            "no function docstring; description borrowed from the module. The "
            "description is what the planner sees -- please improve it."
        )

    arg_docs = _arg_descriptions(docstring)

    args = func.args
    positional = args.posonlyargs + args.args
    defaults: Sequence[Optional[ast.expr]] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)

    inputs: List[ParamDraft] = []
    for arg, default_node in zip(positional, defaults):
        if arg.arg in ("self", "cls"):
            continue
        if arg.arg == "output_dir":
            # Supplied by the runtime, not a planner-visible parameter.
            continue
        annotation = _annotation_text(arg.annotation)
        default_value, literal = _literal_default(default_node)
        if default_node is not None and not literal:
            notes.append(
                f"input '{arg.arg}': default expression could not be read "
                "statically; treated as no default"
            )
        param_type, type_notes = _param_type_for(arg.arg, annotation, default_value)
        notes.extend(type_notes)
        required = default_node is None
        inputs.append(
            ParamDraft(
                name=arg.arg,
                type=cast(ParamType, param_type),
                required=required,
                default=None if required else default_value,
                description=arg_docs.get(arg.arg, ""),
            )
        )

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        if arg.arg == "output_dir":
            continue
        annotation = _annotation_text(arg.annotation)
        default_value, literal = _literal_default(default_node)
        if default_node is not None and not literal:
            notes.append(
                f"input '{arg.arg}': default expression could not be read "
                "statically; treated as no default"
            )
        param_type, type_notes = _param_type_for(arg.arg, annotation, default_value)
        notes.extend(type_notes)
        required = default_node is None
        inputs.append(
            ParamDraft(
                name=arg.arg,
                type=cast(ParamType, param_type),
                required=required,
                default=None if required else default_value,
                description=arg_docs.get(arg.arg, ""),
            )
        )

    if args.vararg or args.kwarg:
        notes.append(
            "function accepts *args/**kwargs; only explicitly named parameters "
            "are exposed to the planner"
        )

    outputs, output_notes = _infer_outputs(func, docstring)
    notes.extend(output_notes)

    subagent, subagent_notes = _suggest_subagent(func.name, module_path, outputs)
    notes.extend(subagent_notes)

    draft = CapabilityDraft(
        tool_name=_tool_name_for(func.name),
        description=description,
        category=category_for_subagent(subagent),
        subagent=subagent,
        source_module=module_path,
        source_function=func.name,
        inputs=inputs,
        outputs=outputs,
    )
    return draft, notes
