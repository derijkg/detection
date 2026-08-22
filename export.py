
#!/usr/bin/env python3

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None

# Directories to exclude from both tree view and file content dump
EXCLUDE_DIR_NAMES: Set[str] = {
    ".git", ".venv", "venv", "env", "ENV", "__pycache__",
    ".vscode", ".idea", "output", "output_new", "output_tmp",
    "data_static", "notebooks", "checkpoints_tmp",
    "checkpoints", "best_model", ".ipynb_checkpoints",
    "wandb", "evals", "old", ".pytest_cache", "htmlcov", "experiments",
    "node_modules", "tests"
}

# Dynamic directory prefixes to ignore
EXCLUDE_DIR_PREFIXES: tuple = (
    ".tmp_optuna",
    ".tmp_",
    "__pycache__"
)

# File extensions to include in code dump
INCLUDE_EXTENSIONS: Set[str] = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".toml", ".sql"}

# Specific file names to ignore
EXCLUDE_FILES: Set[str] = {
    "codebase.md",
    "codebase.xml",
    "codebase.json",
    "codebase.yaml",
    "codebase_linkage.md",
    "codebase_linkage.json",
    "codebase_linkage.xml",
    "codebase_linkage.yaml",
    "evaluation_summary.json",
    ".DS_Store",
    "README.md",
    "TODO.md",
    "export.py",
    "profile_data.py",
}

# File prefixes to ignore
EXCLUDE_FILE_PREFIXES: tuple = (
    "codebase_",
)

# Common boilerplate constants to filter out of compact view
IGNORED_CONSTANTS: Set[str] = {
    "PROJECT_ROOT", "PROJECT_DIR", "ROOT_DIR", "CACHE_ROOT", "CACHE_DIR",
    "INPUT_PATH", "OUTPUT_DIR", "OUTPUT_ROOT", "DEFAULT_LOG_DIR",
    "DEFAULT_DATA_PATH", "DEFAULT_FEATURES_DIR", "__all__"
}


def is_excluded_dir(name: str) -> bool:
    if name in EXCLUDE_DIR_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in EXCLUDE_DIR_PREFIXES)


def is_excluded_file(name: str) -> bool:
    if name in EXCLUDE_FILES:
        return True
    return any(name.startswith(prefix) for prefix in EXCLUDE_FILE_PREFIXES)


def is_excluded_path(path: Path) -> bool:
    return any(is_excluded_dir(part) for part in path.parts)


def get_all_valid_files(root_dir: Path) -> List[Path]:
    valid_files = []
    for path in sorted(root_dir.rglob("*")):
        if path.is_dir() or is_excluded_path(path.relative_to(root_dir)):
            continue
        if path.suffix not in INCLUDE_EXTENSIONS or is_excluded_file(path.name):
            continue
        valid_files.append(path)
    return valid_files


# ==============================================================================
# AST-BASED IMPORT RESOLVER & GRAPH BUILDER
# ==============================================================================

def get_module_name_map(root_dir: Path, py_files: List[Path]) -> Dict[str, Path]:
    mod_map = {}
    for p in py_files:
        rel = p.relative_to(root_dir)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            mod_name = ".".join(parts)
            mod_map[mod_name] = p

        if len(parts) > 1 and parts[0] in {"src", "lib", "app"}:
            sub_name = ".".join(parts[1:])
            mod_map[sub_name] = p

    return mod_map


def resolve_import_path(
    import_entry: Tuple[str, int],
    current_file: Path,
    root_dir: Path,
    module_map: Dict[str, Path]
) -> Optional[Path]:
    mod_name, level = import_entry

    if level > 0:
        current_dir = current_file.parent
        for _ in range(level - 1):
            if current_dir != root_dir:
                current_dir = current_dir.parent

        target_path = current_dir / (mod_name.replace(".", "/") + ".py") if mod_name else current_dir / "__init__.py"
        if target_path.exists() and target_path.is_file():
            return target_path

        target_dir = current_dir / mod_name.replace(".", "/")
        if (target_dir / "__init__.py").exists():
            return target_dir / "__init__.py"
        return None

    if mod_name in module_map:
        return module_map[mod_name]

    parts = mod_name.split(".")
    while len(parts) > 1:
        parts.pop()
        sub_mod = ".".join(parts)
        if sub_mod in module_map:
            return module_map[sub_mod]

    return None


def get_clean_function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    pos_args = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
    num_defaults = len(node.args.defaults)
    default_offset = len(pos_args) - num_defaults

    arg_strs = []
    for idx, arg in enumerate(pos_args):
        if arg.arg in {"self", "cls"}:
            continue
        name = arg.arg
        if idx >= default_offset:
            default_node = node.args.defaults[idx - default_offset]
            try:
                def_val = ast.unparse(default_node)
                name += f"={def_val}"
            except Exception:
                pass
        arg_strs.append(name)

    if node.args.vararg:
        arg_strs.append(f"*{node.args.vararg.arg}")

    for idx, arg in enumerate(node.args.kwonlyargs):
        name = arg.arg
        if idx < len(node.args.kw_defaults) and node.args.kw_defaults[idx] is not None:
            try:
                def_val = ast.unparse(node.args.kw_defaults[idx])
                name += f"={def_val}"
            except Exception:
                pass
        arg_strs.append(name)

    if node.args.kwarg:
        arg_strs.append(f"**{node.args.kwarg.arg}")

    return ", ".join(arg_strs)


def parse_compact_signatures(py_file: Path) -> dict:
    try:
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
    except Exception:
        return {"classes": [], "functions": [], "constants": []}

    classes = []
    functions = []
    constants = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "args": get_clean_function_args(node)
            })

        elif isinstance(node, ast.ClassDef):
            bases_str = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            methods = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": sub.name,
                        "args": get_clean_function_args(sub)
                    })
            classes.append({
                "name": node.name,
                "bases": bases_str,
                "methods": methods
            })

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper() and target.id not in IGNORED_CONSTANTS:
                    constants.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper() and node.target.id not in IGNORED_CONSTANTS:
                constants.append(node.target.id)

    return {
        "classes": classes,
        "functions": functions,
        "constants": sorted(set(constants))
    }


def build_linkage_graph(root_dir: Path, all_py_files: List[Path]):
    module_map = get_module_name_map(root_dir, all_py_files)
    signatures = {}
    forward_deps: Dict[Path, Dict[Path, Set[str]]] = {f: {} for f in all_py_files}
    reverse_deps: Dict[Path, Dict[Path, Set[str]]] = {f: {} for f in all_py_files}

    for py_file in all_py_files:
        signatures[py_file] = parse_compact_signatures(py_file)
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = resolve_import_path((alias.name, 0), py_file, root_dir, module_map)
                    if target and target in forward_deps and target != py_file and target.name != "__init__.py" and py_file.name != "__init__.py":
                        sym_name = alias.asname or alias.name
                        forward_deps[py_file].setdefault(target, set()).add(sym_name)
                        reverse_deps[target].setdefault(py_file, set()).add(sym_name)

            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                level = node.level
                for alias in node.names:
                    target = resolve_import_path((mod, level), py_file, root_dir, module_map)
                    sym_name = alias.name
                    if not target:
                        comb = f"{mod}.{alias.name}" if mod else alias.name
                        target = resolve_import_path((comb, level), py_file, root_dir, module_map)
                        sym_name = "*" if target else alias.name

                    if target and target in forward_deps and target != py_file and target.name != "__init__.py" and py_file.name != "__init__.py":
                        forward_deps[py_file].setdefault(target, set()).add(sym_name)
                        reverse_deps[target].setdefault(py_file, set()).add(sym_name)

    return signatures, forward_deps, reverse_deps


# ==============================================================================
# EXPERIMENT TRACKING & PIPELINE TRACER
# ==============================================================================

def load_experiments_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}

    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            pass

    experiments = {}
    current_name = None
    for line in text.splitlines():
        name_match = re.search(r"^\s*-\s*name:\s*['\"]?([^'\"\s]+)['\"]?", line)
        if name_match:
            current_name = name_match.group(1)
            experiments[current_name] = {"name": current_name}
        elif current_name and ":" in line:
            k, v = line.split(":", 1)
            experiments[current_name][k.strip()] = v.strip()
    return {"experiments": experiments}


def get_available_experiments(config_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    exps = config_data.get("experiments", {})
    if isinstance(exps, list):
        out = {}
        for item in exps:
            if isinstance(item, dict) and "name" in item:
                out[item["name"]] = item
        return out
    elif isinstance(exps, dict):
        return exps
    return {}


def trace_single_experiment(
    exp_name: str,
    exp_cfg: Dict[str, Any],
    root_dir: Path,
    all_files: List[Path],
    forward_deps: Dict[Path, Dict[Path, Set[str]]]
) -> Tuple[Set[Path], List[Dict[str, str]]]:
    file_map = {str(f.relative_to(root_dir)): f for f in all_files}
    active_files: Set[Path] = set()
    stages: List[Dict[str, str]] = []

    def add_rel(rel_str: str) -> Optional[Path]:
        p = file_map.get(rel_str)
        if p:
            active_files.add(p)
            return p
        return None

    # 1. Orchestration & Config Stage
    add_rel("configs/experiments.yaml")
    add_rel("scripts/run_experiment.py")
    add_rel("src/utils/config.py")
    add_rel("src/utils/seed.py")
    add_rel("src/utils/manifest.py")
    add_rel("src/utils/logger.py")
    stages.append({
        "stage": "1. Orchestration & Configuration",
        "detail": "`scripts/run_experiment.py` ➔ loads `src/utils/config.py`, sets `src/utils/seed.py`, inits `src/utils/manifest.py` (`RunTracker`)"
    })

    # 2. Data Pipeline Stage
    add_rel("src/data/data_loader.py")
    add_rel("src/data/dataset_recipe.py")
    add_rel("src/data/stylometrics.py")
    recipe_str = str(exp_cfg.get("recipe", "")).lower() + str(exp_cfg.get("dataset_recipe", "")).lower()
    if "synth" in recipe_str or "synthetic" in recipe_str:
        add_rel("src/data/synth_data.py")
        synth_note = " ➔ generates synthetic data with `src/data/synth_data.py`"
    else:
        synth_note = ""

    stages.append({
        "stage": "2. Data Ingestion & Dataset Recipe",
        "detail": f"`src/data/dataset_recipe.py` (`RecipeDataBuilder`) ➔ queries `src/data/data_loader.py` (`DetectionDataManager`) ➔ normalizes text with `src/data/stylometrics.py`{synth_note}"
    })

    # 3. Model & Training/Tuning Stage
    cfg_text = json.dumps(exp_cfg).lower()
    model_type = str(exp_cfg.get("model_type", exp_cfg.get("model", ""))).lower()
    tuning_enabled = exp_cfg.get("tuning", {}).get("enabled", True) if isinstance(exp_cfg.get("tuning"), dict) else True

    if any(k in model_type or k in cfg_text for k in ["deberta", "mdeberta"]):
        add_rel("src/models/base.py")
        add_rel("src/models/deberta.py")
        add_rel("src/training/trainer_deberta.py")
        if tuning_enabled:
            add_rel("src/training/tune_deberta.py")
            add_rel("src/utils/optuna_utils.py")
            tune_desc = "`src/training/tune_deberta.py` (`DebertaOptunaTuner`)"
        else:
            tune_desc = "Standard trainer"

        stages.append({
            "stage": "3. Model & Hyperparameter Tuning",
            "detail": f"{tune_desc} ➔ trains `src/models/deberta.py` (`MDeBERTaDetector`) using `src/training/trainer_deberta.py` (`build_deberta_trainer`, `RockafellarUryasevCVaRLoss`)"
        })

    elif "svm" in model_type or "svm" in cfg_text:
        add_rel("src/models/base.py")
        add_rel("src/models/svm_pipeline.py")
        if tuning_enabled:
            add_rel("src/training/tune_svm.py")
            add_rel("src/utils/optuna_utils.py")
            tune_desc = "`src/training/tune_svm.py` (`SVMOptunaTuner`)"
        else:
            tune_desc = "`src/models/svm_pipeline.py` (`SVMPipelineFactory`)"

        stages.append({
            "stage": "3. Model & Hyperparameter Tuning",
            "detail": f"{tune_desc} ➔ fits `src/models/svm_pipeline.py` (`SVMDetector`) using stylometric & TF-IDF extractors"
        })

    elif "fast_detect" in model_type or "fast_detect" in cfg_text:
        add_rel("src/models/base.py")
        add_rel("src/models/fast_detect_gpt.py")
        stages.append({
            "stage": "3. Model & Discrepancy Scoring",
            "detail": "`src/models/fast_detect_gpt.py` (`FastDetectGPTDetector`) ➔ loads causal reference models and computes sampling discrepancy"
        })

    elif "statistical" in model_type or "trajectory" in model_type or "statistical" in cfg_text:
        add_rel("src/models/base.py")
        add_rel("src/models/statistical_features.py")
        add_rel("src/models/statistical_detector.py")
        stages.append({
            "stage": "3. Feature Extraction & Classifier",
            "detail": "`src/models/statistical_features.py` ➔ extracts thermodynamic features for `src/models/statistical_detector.py` (`StatisticalTrajectoryDetector`)"
        })
    else:
        add_rel("src/models/base.py")
        add_rel("src/models/registry.py")

    add_rel("src/models/registry.py")

    # 4. Evaluation Stage
    add_rel("src/evaluation/benchmark.py")
    add_rel("src/evaluation/metrics.py")
    stages.append({
        "stage": "4. Evaluation & Calibration",
        "detail": "`src/evaluation/benchmark.py` (`BenchmarkOrchestrator`) ➔ computes PAUC & conformal thresholds in `src/evaluation/metrics.py` (`MetricEvaluator`)"
    })

    # 5. Manifest & Visualization Stage
    add_rel("src/visualization/latex_tables.py")
    add_rel("src/visualization/plots.py")
    stages.append({
        "stage": "5. Manifest Logging & Visualization",
        "detail": "Logs run to `src/utils/manifest.py` (`RunTracker`) ➔ exports LaTeX tables via `src/visualization/latex_tables.py` and plots via `src/visualization/plots.py`"
    })

    # Expand active files with any transitive upstream dependencies
    queue = list(active_files)
    visited = set(active_files)
    while queue:
        curr = queue.pop(0)
        for dep in forward_deps.get(curr, {}):
            if dep not in visited:
                visited.add(dep)
                active_files.add(dep)
                queue.append(dep)

    return active_files, stages


def trace_experiments_pipeline(
    exp_queries: List[str],
    config_path: Path,
    root_dir: Path,
    all_files: List[Path],
    forward_deps: Dict[Path, Dict[Path, Set[str]]]
) -> Tuple[Set[Path], List[Dict[str, Any]]]:
    config_data = load_experiments_yaml(config_path)
    avail = get_available_experiments(config_data)

    if not avail:
        print(f"⚠️  No experiments found in `{config_path.name}`.")
        return set(), []

    all_active_files: Set[Path] = set()
    experiment_traces: List[Dict[str, Any]] = []

    for query in exp_queries:
        matched_name = None
        for name in avail:
            if query.lower() == name.lower() or query.lower() in name.lower():
                matched_name = name
                break

        if not matched_name:
            print(f"⚠️  Experiment '{query}' not found in `{config_path.name}`.")
            continue

        exp_cfg = avail[matched_name]
        files, stages = trace_single_experiment(matched_name, exp_cfg, root_dir, all_files, forward_deps)
        all_active_files.update(files)
        experiment_traces.append({
            "name": matched_name,
            "config": exp_cfg,
            "stages": stages
        })

    return all_active_files, experiment_traces


# ==============================================================================
# GRAPH-POWERED SEARCH / SECTION RESOLUTION
# ==============================================================================

def resolve_section_files_via_graph(
    query: str,
    root_dir: Path,
    all_files: List[Path],
    signatures: Dict[Path, dict],
    forward_deps: Dict[Path, Dict[Path, Set[str]]],
    reverse_deps: Dict[Path, Dict[Path, Set[str]]],
    include_blast_radius: bool = True
) -> Set[Path]:
    query_lower = query.lower()
    seed_files = set()

    for p in all_files:
        rel_str = str(p.relative_to(root_dir)).lower()
        if query_lower in rel_str:
            seed_files.add(p)
            continue

        if p.suffix == ".py":
            sig = signatures.get(p, {})
            if any(query_lower in cls["name"].lower() for cls in sig.get("classes", [])):
                seed_files.add(p)
                continue
            if any(query_lower in fn["name"].lower() for fn in sig.get("functions", [])):
                seed_files.add(p)
                continue
            if any(query_lower in const.lower() for const in sig.get("constants", [])):
                seed_files.add(p)
                continue

    if not seed_files:
        print(f"⚠️  No files or defined symbols matched query: '{query}'")
        return set()

    resolved_files = set(seed_files)

    # 1. Upstream dependencies
    queue = [f for f in seed_files if f.suffix == ".py"]
    visited_deps = set(queue)
    while queue:
        curr = queue.pop(0)
        for dep in forward_deps.get(curr, {}):
            if dep not in visited_deps:
                visited_deps.add(dep)
                resolved_files.add(dep)
                queue.append(dep)

    # 2. Downstream blast radius
    if include_blast_radius:
        queue = [f for f in seed_files if f.suffix == ".py"]
        visited_impacts = set(queue)
        while queue:
            curr = queue.pop(0)
            for dep in reverse_deps.get(curr, {}):
                if dep not in visited_impacts:
                    visited_impacts.add(dep)
                    resolved_files.add(dep)
                    queue.append(dep)

    return resolved_files


# ==============================================================================
# RENDERING
# ==============================================================================

def render_compact_markdown(
    files_to_export: List[Path],
    root_dir: Path,
    signatures: dict,
    forward_deps: dict,
    reverse_deps: dict,
    section_name: Optional[str] = None,
    experiment_traces: Optional[List[Dict[str, Any]]] = None
) -> str:
    lines = []

    if experiment_traces:
        lines.append("# Codebase Context: Experiment Pipeline Trace\n")
        for trace in experiment_traces:
            exp_name = trace["name"]
            exp_cfg = trace["config"]
            model_type = exp_cfg.get("model_type", exp_cfg.get("model", "auto"))
            lines.append(f"## Experiment Pipeline: `{exp_name}`")
            lines.append(f"> **Config Source:** `configs/experiments.yaml` | **Model Type:** `{model_type}`\n")
            lines.append("### Pipeline Execution Stages")
            for stage_info in trace["stages"]:
                lines.append(f"- **{stage_info['stage']}**:")
                lines.append(f"  {stage_info['detail']}")
            lines.append("")
        lines.append("## Active Pipeline Subsystem Linkage & Blast Radius\n")
    else:
        sec_info = f" (Section: `{section_name}`)" if section_name else ""
        lines.append(f"# Codebase Linkage & Blast Radius Graph{sec_info}")
        lines.append("> **Format:** `[File]` ➔ **def** (symbols defined) | **imports** | **impacts** (files directly broken if this file changes)\n")

    for path in sorted(files_to_export):
        if path.suffix != ".py" or path.name == "__init__.py":
            continue

        rel_path = path.relative_to(root_dir)
        sig = signatures.get(path, {})
        f_deps = forward_deps.get(path, {})
        r_deps = reverse_deps.get(path, {})

        consts = sig.get("constants", [])
        funcs = sig.get("functions", [])
        classes = sig.get("classes", [])

        has_defs = bool(consts or funcs or classes)
        has_imports = bool(f_deps)
        has_impacts = bool(r_deps)

        if not has_defs and not has_imports and not has_impacts:
            continue

        lines.append(f"### `{rel_path}`")

        if has_defs:
            if funcs and not classes and not consts:
                fn_strs = [f"`{fn['name']}({fn['args']})`" for fn in funcs]
                lines.append(f"- **def**: {', '.join(fn_strs)}")
            else:
                lines.append("- **def**:")
                if consts:
                    lines.append(f"  - `const`: {', '.join(consts)}")
                if funcs:
                    fn_strs = [f"`{fn['name']}({fn['args']})`" for fn in funcs]
                    lines.append(f"  - `fn`: {', '.join(fn_strs)}")
                for cls in classes:
                    bases_str = f"({cls['bases']})" if cls["bases"] else ""
                    if cls["methods"]:
                        method_strs = [f"{m['name']}({m['args']})" for m in cls["methods"]]
                        lines.append(f"  - `class {cls['name']}{bases_str}`: {', '.join(method_strs)}")
                    else:
                        lines.append(f"  - `class {cls['name']}{bases_str}`")

        if has_imports:
            lines.append("- **imports**:")
            for target_path, syms in sorted(f_deps.items(), key=lambda x: str(x[0])):
                target_rel = target_path.relative_to(root_dir)
                sym_list = sorted(syms)
                sym_str = "[*]" if sym_list == ["*"] else f"[{', '.join(sym_list)}]"
                lines.append(f"  - `{target_rel}`: `{sym_str}`")

        if has_impacts:
            lines.append("- **impacts ➔**")
            for dep_path, syms in sorted(r_deps.items(), key=lambda x: str(x[0])):
                dep_rel = dep_path.relative_to(root_dir)
                sym_list = sorted(syms)
                sym_str = "[*]" if sym_list == ["*"] else f"[{', '.join(sym_list)}]"
                lines.append(f"  - `{dep_rel}`: `{sym_str}`")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_compact_json(
    files_to_export: List[Path],
    root_dir: Path,
    signatures: dict,
    forward_deps: dict,
    reverse_deps: dict,
    experiment_traces: Optional[List[Dict[str, Any]]] = None
) -> str:
    modules_dict = {}
    for path in sorted(files_to_export):
        if path.suffix != ".py" or path.name == "__init__.py":
            continue
        rel_path = str(path.relative_to(root_dir))
        sig = signatures.get(path, {})
        f_deps = {str(k.relative_to(root_dir)): sorted(v) for k, v in forward_deps.get(path, {}).items() if k.name != "__init__.py"}
        r_deps = {str(k.relative_to(root_dir)): sorted(v) for k, v in reverse_deps.get(path, {}).items() if k.name != "__init__.py"}

        if not sig.get("constants") and not sig.get("functions") and not sig.get("classes") and not f_deps and not r_deps:
            continue

        modules_dict[rel_path] = {
            "def": sig,
            "imports": f_deps,
            "impacts": r_deps
        }

    out_dict = {}
    if experiment_traces:
        out_dict["experiment_pipeline_traces"] = experiment_traces
    out_dict["modules"] = modules_dict

    return json.dumps(out_dict, indent=2)


# ==============================================================================
# AST-BASED CLEAN CODE TRANSFORMATION (CORRUPTION-FREE)
# ==============================================================================

class DocstringStripper(ast.NodeTransformer):
    def _strip_docstring(self, node):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
        return self.generic_visit(node)

    def visit_Module(self, node):
        return self._strip_docstring(node)

    def visit_ClassDef(self, node):
        return self._strip_docstring(node)

    def visit_FunctionDef(self, node):
        return self._strip_docstring(node)

    def visit_AsyncFunctionDef(self, node):
        return self._strip_docstring(node)


def clean_python_code(code: str, strip_comments: bool = True) -> str:
    if not strip_comments:
        return compress_whitespace(code)

    try:
        tree = ast.parse(code)
        transformed = DocstringStripper().visit(tree)
        ast.fix_missing_locations(transformed)
        return ast.unparse(transformed)
    except Exception:
        return compress_whitespace(code)


def compress_whitespace(code: str) -> str:
    lines = [line.rstrip() for line in code.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    cleaned = []
    empty_count = 0
    for line in lines:
        if not line:
            empty_count += 1
            if empty_count <= 1:
                cleaned.append(line)
        else:
            empty_count = 0
            cleaned.append(line)

    return "\n".join(cleaned)


def build_filtered_tree(files: List[Path], root_dir: Path) -> str:
    tree_dict: dict = {}
    for f in sorted(files):
        rel = f.relative_to(root_dir)
        curr = tree_dict
        for part in rel.parts:
            curr = curr.setdefault(part, {})

    lines = ["."]
    def _render(d: dict, prefix: str = ""):
        keys = sorted(d.keys())
        for idx, key in enumerate(keys):
            is_last = (idx == len(keys) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{key}")
            if d[key]:
                sub_prefix = prefix + ("    " if is_last else "│   ")
                _render(d[key], sub_prefix)

    _render(tree_dict)
    return "\n".join(lines)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export token-optimized codebase context, linkage graphs, and experiment pipelines for LLMs.")
    parser.add_argument(
        "-e", "--experiment",
        nargs="+",
        default=None,
        help="Track one or more experiments from configs/experiments.yaml through the codebase pipeline."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include the compact linkage overview / pipeline trace at the top followed by the full source code."
    )
    parser.add_argument(
        "-c", "--compact",
        action="store_true",
        help="Generate only the compact Markdown linkage & impact graph without full code dump."
    )
    parser.add_argument(
        "-s", "--section",
        type=str,
        default=None,
        help="Export files relevant to a file/class/function/constant query + upstream dependencies + downstream blast radius."
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all experiments configured in configs/experiments.yaml."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/experiments.yaml",
        help="Path to experiments YAML config (default: configs/experiments.yaml)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: codebase_linkage[_<tag>].md or codebase[_<tag>].xml/.md)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["markdown", "json", "xml"],
        default=None,
        help="Output representation format (default: 'markdown' for compact & full hybrid mode, 'xml' for standard dump)."
    )
    parser.add_argument(
        "--keep-comments",
        dest="strip_comments",
        action="store_false",
        default=True,
        help="Keep comments and docstrings in full dump mode (stripped by default to save tokens)."
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        help="Omit directory tree structure in full dump mode."
    )
    args = parser.parse_args()

    root_dir = Path(".").resolve()
    config_file_path = root_dir / args.config_file

    if args.list_experiments:
        cfg = load_experiments_yaml(config_file_path)
        exps = get_available_experiments(cfg)
        if not exps:
            print(f"No experiments found in `{args.config_file}`.")
        else:
            print(f"Available Experiments in `{args.config_file}`:")
            for name, exp_data in exps.items():
                m_type = exp_data.get("model_type", exp_data.get("model", "auto"))
                print(f"  • {name} (model: {m_type})")
        return

    all_valid_files = get_all_valid_files(root_dir)
    all_py_files = [f for f in all_valid_files if f.suffix == ".py"]

    # Pre-compute linkage graph
    signatures, forward_deps, reverse_deps = build_linkage_graph(root_dir, all_py_files)

    # Determine default format
    selected_format = args.format or ("markdown" if (args.compact or args.full) else "xml")
    ext_map = {"markdown": "md", "json": "json", "xml": "xml"}
    ext = ext_map[selected_format]

    experiment_traces = None

    # 1. Handle Experiment Pipeline Tracing
    if args.experiment:
        selected_files, experiment_traces = trace_experiments_pipeline(
            args.experiment, config_file_path, root_dir, all_valid_files, forward_deps
        )
        if not selected_files:
            print("No files matched experiment pipeline.")
            return
        files_to_export = sorted(selected_files)
        tag = "_".join(re.sub(r'[^a-zA-Z0-9_-]', '_', e) for e in args.experiment)
        prefix = "codebase_pipeline" if args.compact else ("codebase_exp_full" if args.full else "codebase_exp")
        default_out_name = f"{prefix}_{tag}.{ext}"

    # 2. Handle Symbol/Section Query
    elif args.section:
        selected_files = resolve_section_files_via_graph(
            args.section, root_dir, all_valid_files, signatures, forward_deps, reverse_deps
        )
        if not selected_files:
            print("No files to export.")
            return
        files_to_export = sorted(selected_files)
        tag = re.sub(r'[^a-zA-Z0-9_-]', '_', args.section)
        prefix = "codebase_linkage" if args.compact else ("codebase_full" if args.full else "codebase")
        default_out_name = f"{prefix}_{tag}.{ext}"

    # 3. Whole Codebase
    else:
        files_to_export = all_valid_files
        prefix = "codebase_linkage" if args.compact else ("codebase_full" if args.full else "codebase")
        default_out_name = f"{prefix}.{ext}"

    output_path = Path(args.output) if args.output else Path(default_out_name)

    # --------------------------------------------------------------------------
    # 1. ONLY COMPACT MODE
    # --------------------------------------------------------------------------
    if args.compact and not args.full:
        if selected_format == "json":
            content = render_compact_json(
                files_to_export, root_dir, signatures, forward_deps, reverse_deps, experiment_traces=experiment_traces
            )
        else:
            content = render_compact_markdown(
                files_to_export, root_dir, signatures, forward_deps, reverse_deps,
                section_name=args.section, experiment_traces=experiment_traces
            )

        output_path.write_text(content, encoding="utf-8")
        print(f"Exported compact linkage graph for {len(files_to_export)} files to `{output_path.name}`.")
        return

    # --------------------------------------------------------------------------
    # 2. FULL DUMP (OR HYBRID --full WITH OVERVIEW ON TOP)
    # --------------------------------------------------------------------------
    with open(output_path, "w", encoding="utf-8") as out:
        if selected_format == "xml":
            out.write("<codebase>\n")

            # In XML mode with --full, prepend the linkage map
            if args.full:
                out.write("  <linkage_overview>\n")
                out.write(render_compact_markdown(
                    files_to_export, root_dir, signatures, forward_deps, reverse_deps,
                    section_name=args.section, experiment_traces=experiment_traces
                ))
                out.write("\n  </linkage_overview>\n\n")

            if not args.no_tree:
                out.write("  <file_tree>\n")
                out.write(build_filtered_tree(files_to_export, root_dir))
                out.write("\n  </file_tree>\n\n")

            out.write("  <files>\n")
            for path in files_to_export:
                rel_path = path.relative_to(root_dir)
                try:
                    raw_code = path.read_text(encoding="utf-8")
                    if path.suffix == ".py":
                        content = clean_python_code(raw_code, strip_comments=args.strip_comments)
                    else:
                        content = compress_whitespace(raw_code)
                except Exception as e:
                    content = f"Error reading file: {e}"

                out.write(f'    <file path="{rel_path}">\n')
                out.write(content)
                out.write("\n    </file>\n")
            out.write("  </files>\n")
            out.write("</codebase>\n")

        else:
            # Markdown output
            if args.full:
                # Prepend the full compact overview / experiment pipeline trace on top!
                overview_text = render_compact_markdown(
                    files_to_export, root_dir, signatures, forward_deps, reverse_deps,
                    section_name=args.section, experiment_traces=experiment_traces
                )
                out.write(overview_text)
                out.write("\n---\n\n")

            header_tag = f" (Experiment: {', '.join(args.experiment)})" if args.experiment else (f" (Section: {args.section})" if args.section else "")
            out.write(f"# Implementation Source Files{header_tag}\n\n")

            if not args.no_tree:
                out.write("## Structure\n```text\n")
                out.write(build_filtered_tree(files_to_export, root_dir))
                out.write("\n```\n\n")

            out.write("## Files\n\n")
            for path in files_to_export:
                rel_path = path.relative_to(root_dir)
                lang = path.suffix.lstrip(".")
                if lang == "sh":
                    lang = "bash"
                elif lang == "yml":
                    lang = "yaml"

                try:
                    raw_code = path.read_text(encoding="utf-8")
                    if path.suffix == ".py":
                        content = clean_python_code(raw_code, strip_comments=args.strip_comments)
                    else:
                        content = compress_whitespace(raw_code)
                except Exception as e:
                    content = f"Error reading file: {e}"

                fence = "````" if "```" in content else "```"
                out.write(f"### `{rel_path}`\n")
                out.write(f"{fence}{lang}\n")
                out.write(content)
                out.write(f"\n{fence}\n\n")

    print(f"Exported {'hybrid' if args.full else 'full'} context ({len(files_to_export)} files) to `{output_path.name}`.")


if __name__ == "__main__":
    main()