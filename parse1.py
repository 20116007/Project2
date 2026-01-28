import importlib
import json
import os
from pathlib import Path
import re
import sys

# Load clang and its cindex module dynamically so linters don't flag missing imports
clang = importlib.import_module("clang")
importlib.import_module("clang.cindex")

# --- Simple .env loader (no external deps) ---
def load_dotenv(dotenv_path):
    try:
        content = Path(dotenv_path).read_text(encoding="utf-8")
    except Exception:
        return
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

load_dotenv(Path(__file__).parent / ".env")

# ---- Helper Functions for Auto detecting Include Paths----

def get_include_paths_from_files(c_files):
    """
    Scans a list of C files for #include "..." directives
    and returns a set of unique parent directories
    """
    include_dirs = set()
    local_include_regex = re.compile(r'#\s*include\s*["<]([^">]+)[">]')

    print("\n--- Auto-detecting include paths ---") 
    for c_file in c_files:
        try:
            content = c_file.read_text(encoding = 'utf-8', errors = 'replace')
            for match in local_include_regex.finditer(content):
                rel_header = match.group(1)
                #resolve the header path relative to the C file
                header_path = (c_file.parent / rel_header).resolve()
                if header_path.exists():
                    include_dirs.add(str(header_path.parent))
                    print(f"     [+]Found: {header_path.parent} (from {rel_header} in {c_file.name})")
        except Exception as e:
            print(f"     [!] Warning Scanning {c_file}: {e}")
    print(f"--- Detection complete. Found {len(include_dirs)} unique include directories ---\n")
    return include_dirs

# --- Main parsing script ---

# Initialize the libclang index
index = clang.cindex.Index.create()

# 1. Get Project folder from user
root_dir = Path(__file__).parent
folder_name = input('Enter folder name (e.g., apl001):\t').strip()
project_path = root_dir / f'src/{folder_name}'

if not project_path.exists():
    print(f"Error: The folder '{project_path}' does not exist.")
    sys.exit(1)

# 2. Find all C source files in the project

c_files = list(project_path.glob('**/*.c'))
if not c_files:
    print(f"No C source files found in '{project_path}'.")
    sys.exit(1)
print(f"Found {len(c_files)} C source files:  {[f.name for f in c_files]}")

#3 Auto-detect include paths using the helper function
include_dirs  = get_include_paths_from_files(c_files)

#4. Add standard fallback directories (as strings to avoid Path mix)
include_dirs.add(str(project_path))
include_dirs.add(str(root_dir / 'headers'))
include_dirs.add(str(root_dir / 'header'))

#5. Build the arguments list for libclang
#define NULL to prevent parsing erros
args = ['-DNULL=0']
# Add SDK sysroot if available (helps resolve standard headers on macOS)
sdk_paths = [
    Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"),
    Path("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"),
]
for sdk_path in sdk_paths:
    if sdk_path.exists():
        args.extend(["-isysroot", str(sdk_path)])
        break
# Normalize include dirs to strings and sort to avoid Path/str TypeError
include_dir_args = sorted({str(d) for d in include_dirs})
#Add all detected and fallback include paths
for d in include_dir_args:
    args.append(f'-I{d}')

print(f"\nUsing {len(include_dir_args)} include paths for parsing.")
for d in include_dir_args:
    print(f"  - {d}")
print()

# --- Collect include usage per source file (for visibility gating) ---
include_regex = re.compile(r'#\s*include\s*["<]([^">]+)[">]')
include_search_dirs = [Path(d) for d in include_dir_args]
file_includes = {}  # c_file path -> set of resolved header paths
file_include_basenames = {}  # c_file path -> set of header basenames

def resolve_include(header_name, current_dir, search_dirs):
    # Try relative to current file, then include dirs
    candidates = [current_dir / header_name]
    candidates += [d / header_name for d in search_dirs]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None

for c_file in c_files:
    includes = set()
    basenames = set()
    try:
        content = c_file.read_text(encoding='utf-8', errors='replace')
        for match in include_regex.finditer(content):
            header_name = match.group(1)
            basenames.add(Path(header_name).name)
            resolved = resolve_include(header_name, c_file.parent, include_search_dirs)
            if resolved:
                includes.add(resolved)
    except Exception:
        pass
    file_includes[str(c_file)] = includes
    file_include_basenames[str(c_file)] = basenames

# --- Global State for Multi-FIle Analysis ---
global_definitions = {} # Maps function names to their AST nodes
file_tus = {}           # Maps file paths to their TranslationUnits
macro_map = {}        # Maps macro names to their values
macro_nodes = {}      # Maps macro names to their AST nodes (for future AST use)
header_decl_map = {}   # Maps header paths to function declarations
header_decl_basenames = {}  # Maps header basenames to function declarations

# 6. Parse all c files and populate global maps
print("--- Parsing files and capturing functions/macros...\n")
for c_file in c_files:
    print(f"Parsing {c_file.name}...")
    try:
        tu = index.parse(
            str(c_file),
            args=args,
            options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        if not tu:
            print(f"  [!] Error: Unable to parse {c_file.name}. Skipping.")
            continue
        file_tus[str(c_file)] = tu
        # Surface diagnostics to catch missing includes or parse errors
        for diag in tu.diagnostics:
            print(f"  [diag:{diag.severity}] {diag.spelling} ({diag.location.file}:{diag.location.line})")

        # Capture function definitions
        for node in tu.cursor.get_children():
            if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and node.is_definition():
                global_definitions[node.spelling] = node
        # Capture macro definitions and store raw values for resolution
        for node in tu.cursor.get_children():
            if node.kind == clang.cindex.CursorKind.MACRO_DEFINITION:
                macro_nodes[node.spelling] = node
                tokens = list(node.get_tokens())
                if len(tokens) > 1:
                    value = "".join([t.spelling for t in tokens[1:]])
                    macro_map[node.spelling] = value
                else:
                    macro_map.setdefault(node.spelling, "")

        # Capture function declarations from headers (visibility gating)
        for node in tu.cursor.walk_preorder():
            if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and not node.is_definition():
                loc = node.location.file
                if not loc:
                    continue
                header_path = str(Path(loc.name).resolve())
                header_decl_map.setdefault(header_path, set()).add(node.spelling)
                header_decl_basenames.setdefault(Path(header_path).name, set()).add(node.spelling)
        print(f"  [+] Parsed successfully.")

    except Exception as e:
        print(f"  [!] Exception parsing {c_file.name}: {e}")

print(f"\n--- Prasing complete---")
print(f"Found {len(global_definitions)} functions definitions.")
print(f"Found {len(macro_map)} macro definitions.\n")

# --- AST Serialization Helpers (JSON output) ---

AST_MAX_DEPTH = 6  # Limit depth to keep JSON size manageable; adjust as needed

def cursor_to_dict(node, depth=0):
    if not node:
        return None
    if depth > AST_MAX_DEPTH:
        return {
            "kind": node.kind.name,
            "spelling": node.spelling,
            "displayname": node.displayname,
            "truncated": True,
        }

    location = node.location
    extent = node.extent
    return {
        "kind": node.kind.name,
        "spelling": node.spelling,
        "displayname": node.displayname,
        "location": {
            "file": location.file.name if location.file else None,
            "line": location.line,
            "column": location.column,
        },
        "extent": {
            "start": {
                "line": extent.start.line,
                "column": extent.start.column,
            },
            "end": {
                "line": extent.end.line,
                "column": extent.end.column,
            },
        },
        "children": [cursor_to_dict(child, depth + 1) for child in node.get_children()],
    }

# --- AST JSON Output ---
def build_linkage_map():
    linkage = []
    for func_name, func_node in global_definitions.items():
        func_file = func_node.extent.start.file.name if func_node.extent.start.file else None
        for node in func_node.walk_preorder():
            if node.kind != clang.cindex.CursorKind.CALL_EXPR:
                continue
            callee = get_called_function_name(node)
            if not callee:
                continue
            callee_node = global_definitions.get(callee)
            callee_file = callee_node.extent.start.file.name if callee_node and callee_node.extent.start.file else None
            if func_file and callee_file and is_callee_visible_from_file(callee, func_file):
                linkage.append({
                    "caller": func_name,
                    "caller_file": func_file,
                    "callee": callee,
                    "callee_file": callee_file,
                    "callsite": {
                        "file": node.location.file.name if node.location.file else None,
                        "line": node.location.line if node.location else None,
                    },
                })
    return linkage

ast_json = None

# --- Helper Function for Tracing (Unchanges from your version) ---

def get_source_text(node):
    if not node or not node.extent.start.file:
        return "<no source>"
    file_name = node.extent.start.file.name
    try:
        with open(file_name, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        s, e = node.extent.start, node.extent.end
        # Single-line spans should include the whole node extent
        if s.line == e.line:
            return lines[s.line - 1][s.column - 1:e.column - 1].strip()
        else:
            return ''.join([
                lines[s.line - 1][s.column - 1:],
                *lines[s.line:e.line - 1],
                lines[e.line - 1][:e.column - 1]
            ]).strip()
    except:
        try: 
            tu = file_tus[file_name]
            return ' '.join(t.spelling for t in tu.get_tokens(extent=node.extent))
        except:
            return "<source error>"

def get_node_source(node):
    if not node or not node.extent.start.file:
        return ""
    return get_source_text(node)

def extract_arguments_from_call(call_node):
    children = list(call_node.get_children())
    return children[1:]  if len(children) > 1 else []

def find_node_path(root,target):
    if root == target:
        return [root]
    for child in root.get_children():
        path = find_node_path(child,target)
        if path:
            return [root] + path
    return None

def find_containing_function(call_node):
    tu = file_tus.get(call_node.extent.start.file.name)
    if not tu:
        return None
    path = find_node_path(tu.cursor, call_node)
    if not path:
        return None
    for node in reversed(path):
        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            return node
    return None

def find_return_expression(func_node):
    for node in func_node.get_children():
        if node.kind == clang.cindex.CursorKind.RETURN_STMT:
            children = list(node.get_children())
            return children[0] if children else None
    return None

def find_global_decl_value(var_name, file_name):
    # Resolve file-scope initializers for variables not assigned inside a function
    tu = file_tus.get(file_name)
    if not tu:
        return None
    for node in tu.cursor.get_children():
        if node.kind == clang.cindex.CursorKind.VAR_DECL and node.spelling == var_name:
            children = list(node.get_children())
            if children:
                return children[0]
    return None

def deep_resolve(node, call_node, scope_node, depth = 0 ):
    if depth > 15:
        return "recursion limit reached"
    raw_text = get_source_text(node).strip()

    if node.kind in [clang.cindex.CursorKind.INTEGER_LITERAL,
                    clang.cindex.CursorKind.STRING_LITERAL]:
        return raw_text
    if raw_text in macro_map:
        return f"{macro_map[raw_text]} (Macro: {raw_text})"
    
    # Resolve explicit macro instantiation/expansion nodes
    macro_kinds = []
    if hasattr(clang.cindex.CursorKind, "MACRO_INSTANTIATION"):
        macro_kinds.append(clang.cindex.CursorKind.MACRO_INSTANTIATION)
    if hasattr(clang.cindex.CursorKind, "MACRO_EXPANSION"):
        macro_kinds.append(clang.cindex.CursorKind.MACRO_EXPANSION)
    if macro_kinds and node.kind in macro_kinds:
        if node.spelling in macro_map:
            return f"{macro_map[node.spelling]} (Macro: {node.spelling})"

    if node.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
        var_name = node.spelling

        if var_name in macro_map:
            return f"{macro_map[var_name]} (Macro: {var_name})"
        
        # Resolve local variable value first, then fall back to global initializer
        assign = trace_variable_backwards(var_name, call_node.location.line, scope_node)
        if assign:
            return deep_resolve(assign, call_node, scope_node, depth + 1)

        global_assign = find_global_decl_value(var_name, call_node.location.file.name)
        if global_assign:
            return deep_resolve(global_assign, call_node, scope_node, depth + 1)
        return f"{var_name} (untraced)"
    
    children = list(node.get_children())
    if children and node.kind in [clang.cindex.CursorKind.UNEXPOSED_EXPR,
                                 clang.cindex.CursorKind.PAREN_EXPR]:
        return deep_resolve(children[0], call_node, scope_node, depth + 1)
    
    return raw_text

# --- MCP Tool Implementations (Generalized) ---

def get_function_def(name):
    node = global_definitions.get(name)
    if not node:
        return None
    return {
        "signature": node.displayname,
        "body": get_node_source(node),
        "file": node.extent.start.file.name if node.extent.start.file else None,
    }

def get_macro_value(name):
    if name in macro_map:
        return {
            "value": macro_map[name],
        }
    # Fallback: scan headers in include dirs for a simple #define
    define_re = re.compile(rf'^\s*#\s*define\s+{re.escape(name)}\s+(.*)$')
    for inc_dir in include_search_dirs:
        if not inc_dir.exists():
            continue
        for header in inc_dir.rglob("*.h"):
            try:
                for line in header.read_text(encoding="utf-8", errors="replace").splitlines():
                    m = define_re.match(line)
                    if m:
                        value = m.group(1).strip()
                        value = value.split("/*")[0].split("//")[0].strip()
                        if value:
                            macro_map[name] = value
                            return {"value": value}
            except Exception:
                continue
    return None

def get_var_assignments(name, scope, file_path):
    func_node = global_definitions.get(scope)
    if not func_node:
        return None
    assignments = []
    for node in func_node.walk_preorder():
        if node.location.file and node.location.file.name != file_path:
            continue
        if node.kind == clang.cindex.CursorKind.VAR_DECL and node.spelling == name:
            children = list(node.get_children())
            if children:
                assignments.append(get_node_source(node))
        if node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
            tokens = [t.spelling for t in node.get_tokens()]
            if "=" in tokens:
                children = list(node.get_children())
                if len(children) == 2 and children[0].spelling == name:
                    assignments.append(get_node_source(node))
    return {"assignments": assignments}

def get_callsite_context(callsite_id):
    call_node = callsite_index.get(callsite_id)
    if not call_node:
        return None
    containing = find_containing_function(call_node)
    return {
        "line": get_node_source(call_node),
        "file": call_node.extent.start.file.name if call_node.extent.start.file else None,
        "line_number": call_node.location.line if call_node.location else None,
        "function": containing.spelling if containing else None,
    }

def get_included_headers(file_path):
    return {"headers": sorted(file_includes.get(file_path, set()))}

def get_declared_functions(header_path):
    return {"functions": sorted(header_decl_map.get(header_path, set()))}

def safe_eval_int(expr_text):
    if not expr_text:
        return None
    if not re.fullmatch(r"[0-9a-fA-FxX\\s\\+\\-\\*/%\\(\\)<>\\|&^]+", expr_text):
        return None
    try:
        return int(eval(expr_text, {"__builtins__": {}}, {}))
    except Exception:
        return None

def evaluate_expr(expr):
    value = safe_eval_int(expr)
    return {"value": value} if value is not None else None

MCP_TOOLS = {
    "get_function_def": get_function_def,
    "get_macro_value": get_macro_value,
    "get_var_assignments": get_var_assignments,
    "get_callsite_context": get_callsite_context,
    "get_included_headers": get_included_headers,
    "get_declared_functions": get_declared_functions,
    "evaluate_expr": evaluate_expr,
}

def run_tool(tool_name, **kwargs):
    tool = MCP_TOOLS.get(tool_name)
    if not tool:
        return None
    return tool(**kwargs)

# --- LLM Orchestrator Skeleton (Autonomous Loop) ---

def llm_stub_decide(state):
    # Placeholder: integrate a real LLM here (Groq/local model/etc.)
    return {
        "type": "tool_requests",
        "tool_requests": [],
        "reason": "llm_not_configured",
    }

def llm_groq_decide(state):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "type": "tool_requests",
            "tool_requests": [],
            "reason": "missing_groq_api_key",
        }
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        import requests
    except Exception:
        return {
            "type": "tool_requests",
            "tool_requests": [],
            "reason": "missing_requests_dependency",
        }
    prompt = (
        "You are a resolver. Never guess. If info is missing, request tools. "
        "Respond ONLY in JSON with keys: type, value, tool_requests.\n"
        f"STATE:\n{json.dumps(state, indent=2)}"
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a resolver. Never guess. If info is missing, request tools. Respond ONLY in JSON with keys: type, value, tool_requests."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        return {
            "type": "tool_requests",
            "tool_requests": [],
            "reason": f"groq_http_{resp.status_code}",
        }
    data = resp.json()
    text = None
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = None
    if not text:
        return {
            "type": "tool_requests",
            "tool_requests": [],
            "reason": "groq_empty_response",
        }
    try:
        return json.loads(text)
    except Exception:
        return {
            "type": "tool_requests",
            "tool_requests": [],
            "reason": "groq_invalid_json",
        }

def llm_local_ollama_decide(state):
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    try:
        import requests
    except Exception:
        return llm_stub_decide(state)
    prompt = (
        "You are a resolver. Never guess. If info is missing, request tools. "
        "Respond ONLY in JSON with keys: type, value, tool_requests.\n"
        f"STATE:\n{json.dumps(state, indent=2)}"
    )
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0,
    }
    resp = requests.post(f"{base_url}/api/generate", json=body, timeout=30)
    if resp.status_code != 200:
        return llm_stub_decide(state)
    data = resp.json()
    text = data.get("response")
    if not text:
        return llm_stub_decide(state)
    try:
        return json.loads(text)
    except Exception:
        return llm_stub_decide(state)

class LLMResolverOrchestrator:
    def __init__(self, llm_decider=None, max_steps=8):
        if llm_decider is None:
            llm_decider = llm_groq_decide
        self.llm_decider = llm_decider
        self.max_steps = max_steps

    def resolve(self, target_expr, callsite_id):
        state = {
            "goal": f"Resolve value for: {target_expr}",
            "target_expr": target_expr,
            "callsite_id": callsite_id,
            "facts": {},
            "tool_results": [],
        }

        for _ in range(self.max_steps):
            auto_response = deterministic_autoresolve(state)
            if auto_response.get("type") == "final":
                return {"value": auto_response.get("value"), "reason": "autoresolver"}
            if auto_response.get("type") == "tool_requests":
                response = auto_response
            else:
                response = self.llm_decider(state)
            if response.get("type") == "final":
                return {"value": response.get("value"), "reason": response.get("reason", "llm_final")}
            tool_requests = response.get("tool_requests", [])
            if not tool_requests:
                return {"value": None, "reason": response.get("reason", "no_tool_requests")}
            for req in tool_requests:
                result = run_tool(req.get("tool"), **req.get("args", {}))
                state["tool_results"].append({
                    "tool": req.get("tool"),
                    "args": req.get("args", {}),
                    "result": result,
                })
        return {"value": None, "reason": "max_steps_reached"}

def should_trigger_llm(resolved_value):
    if not resolved_value:
        return False
    text = resolved_value.strip()
    return bool(re.match(r"^[A-Za-z_]\w*\s*\(.*\)$", text))

# --- Deterministic Autoresolver Policy ---

_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_KEYWORDS = {
    "return", "if", "else", "for", "while", "switch", "case",
    "break", "continue", "int", "char", "void", "long", "short",
    "unsigned", "signed", "static", "const", "struct", "typedef",
}

def _extract_identifiers(expr):
    return {m.group(0) for m in _IDENT_RE.finditer(expr) if m.group(0) not in _KEYWORDS}

def _latest_tool_result(state, tool_name, predicate=None):
    for item in reversed(state.get("tool_results", [])):
        if item.get("tool") != tool_name:
            continue
        if predicate and not predicate(item.get("args", {})):
            continue
        return item.get("result")
    return None

def _tool_attempted(state, tool_name, predicate=None):
    for item in reversed(state.get("tool_results", [])):
        if item.get("tool") != tool_name:
            continue
        if predicate and not predicate(item.get("args", {})):
            continue
        return True
    return False

def _parse_assignment_rhs(line, var_name):
    # naive parse: split on first '=' and strip trailing ';'
    if "=" not in line:
        return None
    lhs, rhs = line.split("=", 1)
    if var_name not in lhs:
        return None
    rhs = rhs.strip().rstrip(";")
    return rhs

def _resolve_var_from_assignments(assignments, var_name, known_values):
    # Evaluate assignments in order, carrying forward known values
    current = None
    for line in assignments:
        rhs = _parse_assignment_rhs(line, var_name)
        if not rhs:
            continue
        rhs_sub = _substitute_knowns(rhs, known_values)
        eval_res = evaluate_expr(rhs_sub)
        if eval_res and eval_res.get("value") is not None:
            current = str(eval_res.get("value"))
            known_values[var_name] = current
        else:
            current = rhs_sub
            known_values[var_name] = current
    return current

def _get_function_body_node(func_node):
    for child in func_node.get_children():
        if child.kind == clang.cindex.CursorKind.COMPOUND_STMT:
            return child
    return None

def _eval_simple_expr(node, env):
    if node is None:
        return None
    if node.kind == clang.cindex.CursorKind.INTEGER_LITERAL:
        value = safe_eval_int(get_source_text(node))
        return value
    if node.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
        name = node.spelling
        if name in env:
            return safe_eval_int(str(env[name])) if isinstance(env[name], str) else env[name]
        if name in macro_map:
            return safe_eval_int(macro_map[name])
        return None
    if node.kind == clang.cindex.CursorKind.PAREN_EXPR:
        children = list(node.get_children())
        return _eval_simple_expr(children[0], env) if children else None
    if node.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR:
        children = list(node.get_children())
        return _eval_simple_expr(children[0], env) if children else None
    if node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
        children = list(node.get_children())
        if len(children) != 2:
            return None
        left = _eval_simple_expr(children[0], env)
        right = _eval_simple_expr(children[1], env)
        if left is None or right is None:
            return None
        tokens = [t.spelling for t in node.get_tokens()]
        # handle comparisons and arithmetic
        for op in ["==", "!=", ">=", "<=", ">", "<", "+", "-", "*", "/", "%"]:
            if op in tokens:
                try:
                    if op == "==":
                        return 1 if left == right else 0
                    if op == "!=":
                        return 1 if left != right else 0
                    if op == ">=":
                        return 1 if left >= right else 0
                    if op == "<=":
                        return 1 if left <= right else 0
                    if op == ">":
                        return 1 if left > right else 0
                    if op == "<":
                        return 1 if left < right else 0
                    if op == "+":
                        return left + right
                    if op == "-":
                        return left - right
                    if op == "*":
                        return left * right
                    if op == "/":
                        return int(left / right)
                    if op == "%":
                        return left % right
                except Exception:
                    return None
    return None

def _eval_var_in_function(func_node, var_name, env):
    body = _get_function_body_node(func_node)
    if not body:
        return None

    def walk(node):
        if node.kind == clang.cindex.CursorKind.IF_STMT:
            children = list(node.get_children())
            if len(children) >= 2:
                cond = _eval_simple_expr(children[0], env)
                then_node = children[1]
                else_node = children[2] if len(children) > 2 else None
                branch = then_node if cond else else_node
                if branch:
                    result = walk(branch)
                    if result is not None:
                        return result
            return None
        if node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
            tokens = [t.spelling for t in node.get_tokens()]
            if "=" in tokens:
                children = list(node.get_children())
                if len(children) == 2:
                    lhs = children[0]
                    rhs = children[1]
                    if lhs.kind == clang.cindex.CursorKind.DECL_REF_EXPR and lhs.spelling == var_name:
                        value = _eval_simple_expr(rhs, env)
                        if value is None:
                            value = get_source_text(rhs)
                        env[var_name] = value
                        return value
        if node.kind == clang.cindex.CursorKind.VAR_DECL and node.spelling == var_name:
            children = list(node.get_children())
            if children:
                value = _eval_simple_expr(children[0], env)
                if value is None:
                    value = get_source_text(children[0])
                env[var_name] = value
                return value
        for child in node.get_children():
            result = walk(child)
            if result is not None:
                return result
        return None

    return walk(body)

def _substitute_knowns(expr, known_map):
    out = expr
    for k, v in known_map.items():
        out = re.sub(rf"\b{re.escape(k)}\b", f"({v})", out)
    return out

def deterministic_autoresolve(state):
    target = state.get("target_expr", "")
    callsite_id = state.get("callsite_id")
    callsite = _latest_tool_result(state, "get_callsite_context", lambda a: a.get("callsite_id") == callsite_id)
    if not callsite:
        return {"type": "tool_requests", "tool_requests": [{"tool": "get_callsite_context", "args": {"callsite_id": callsite_id}}]}

    file_path = callsite.get("file")
    scope = callsite.get("function")

    # If target is a function call, try to resolve its return expression.
    m = re.match(r"^([A-Za-z_]\w*)\s*\((.*)\)$", target)
    if m:
        func_name = m.group(1)
        arg_text = m.group(2).strip()
        func_def = _latest_tool_result(state, "get_function_def", lambda a: a.get("name") == func_name)
        if not func_def:
            return {"type": "tool_requests", "tool_requests": [{"tool": "get_function_def", "args": {"name": func_name}}]}

        args = [a.strip() for a in arg_text.split(",")] if arg_text else []
        known_values = {}

        # Try resolving arg expressions directly if they are literals or macros
        for arg in args:
            if re.fullmatch(r"[0-9]+", arg):
                known_values[arg] = arg
            elif _extract_identifiers(arg):
                # If arg is a macro, request it
                for ident in _extract_identifiers(arg):
                    macro = _latest_tool_result(state, "get_macro_value", lambda a: a.get("name") == ident)
                    if not macro:
                        if not _tool_attempted(state, "get_macro_value", lambda a: a.get("name") == ident):
                            return {"type": "tool_requests", "tool_requests": [{"tool": "get_macro_value", "args": {"name": ident}}]}
                        continue
                    if macro.get("value"):
                        known_values[ident] = macro.get("value")

        for arg in args:
            # request assignments for variables
            if _latest_tool_result(state, "get_var_assignments", lambda a: a.get("name") == arg and a.get("scope") == scope):
                assignments = _latest_tool_result(state, "get_var_assignments", lambda a: a.get("name") == arg and a.get("scope") == scope)
                if assignments:
                    _resolve_var_from_assignments(assignments.get("assignments", []), arg, known_values)
            else:
                if not _tool_attempted(state, "get_var_assignments", lambda a: a.get("name") == arg and a.get("scope") == scope):
                    return {"type": "tool_requests", "tool_requests": [{"tool": "get_var_assignments", "args": {"name": arg, "scope": scope, "file_path": file_path}}]}

        # Extract first return expression
        body = func_def.get("body", "")
        ret_match = re.search(r"return\s+([^;]+);", body)
        if ret_match:
            ret_expr = ret_match.group(1).strip()
            # substitute arguments
            params = func_def.get("signature", "")
            param_list = []
            if "(" in params and ")" in params:
                param_text = params.split("(", 1)[1].rsplit(")", 1)[0]
                param_list = [p.strip().split()[-1] for p in param_text.split(",") if p.strip()]
            for arg_name, arg_val in zip(param_list, args):
                ret_expr = re.sub(rf"\b{re.escape(arg_name)}\b", arg_val, ret_expr)
            ret_expr = _substitute_knowns(ret_expr, known_values)
            # resolve identifiers inside return expression
            for ident in _extract_identifiers(ret_expr):
                if ident in param_list or ident in known_values:
                    continue
                # try local assignment inside callee
                callee_file = func_def.get("file")
                # Prefer AST-based evaluation with known parameter values
                func_node = global_definitions.get(func_name)
                if func_node:
                    env = dict(known_values)
                    rhs_val = _eval_var_in_function(func_node, ident, env)
                    if rhs_val is not None:
                        ret_expr = re.sub(rf"\b{re.escape(ident)}\b", f"({rhs_val})", ret_expr)
                        continue
                assign_res = _latest_tool_result(
                    state,
                    "get_var_assignments",
                    lambda a: a.get("name") == ident and a.get("scope") == func_name,
                )
                if not assign_res:
                    if not _tool_attempted(state, "get_var_assignments", lambda a: a.get("name") == ident and a.get("scope") == func_name):
                        return {"type": "tool_requests", "tool_requests": [{"tool": "get_var_assignments", "args": {"name": ident, "scope": func_name, "file_path": callee_file}}]}
                    assign_res = None
                last = assign_res.get("assignments", [])[-1] if assign_res.get("assignments") else None
                if last:
                    rhs = _parse_assignment_rhs(last, ident)
                    if rhs:
                        ret_expr = re.sub(rf"\b{re.escape(ident)}\b", f"({rhs})", ret_expr)
                        continue
                # try macro
                macro = _latest_tool_result(state, "get_macro_value", lambda a: a.get("name") == ident)
                if macro and macro.get("value"):
                    ret_expr = re.sub(rf"\b{re.escape(ident)}\b", f"({macro.get('value')})", ret_expr)
                else:
                    if not _tool_attempted(state, "get_macro_value", lambda a: a.get("name") == ident):
                        return {"type": "tool_requests", "tool_requests": [{"tool": "get_macro_value", "args": {"name": ident}}]}
                    # leave as symbolic if already attempted
                    continue
            eval_res = evaluate_expr(ret_expr)
            if eval_res and eval_res.get("value") is not None:
                return {"type": "final", "value": str(eval_res.get("value"))}
            return {"type": "final", "value": ret_expr}

        return {"type": "final", "value": target}

    return {"type": "tool_requests", "tool_requests": []}

def trace_variable_backwards(var_name, call_line, scope_node):
    # Walk function scope and capture the latest assignment before the call site
    last_assignment = None
    for node in scope_node.walk_preorder():
        if not node.location.file or node.location.line >= call_line:
            continue

        if node.kind == clang.cindex.CursorKind.VAR_DECL and node.spelling == var_name:
            children = list(node.get_children())
            if children:
                last_assignment = children[0]
            continue

        if node.kind == clang.cindex.CursorKind.BINARY_OPERATOR:
            children = list(node.get_children())
            if len(children) == 2 and children[0].spelling == var_name:
                tokens = [t.spelling for t in node.get_tokens()]
                if '=' in tokens:
                    last_assignment = children[1]
    return last_assignment

# --- Find and Trace Target Function Calls ---

def get_called_function_name(call_node):
    # Extract callee spelling from a CALL_EXPR node
    if call_node.spelling:
        return call_node.spelling
    if call_node.displayname:
        return call_node.displayname.split("(")[0]

    def unwrap_callee(node):
        # Skip wrapper nodes like UNEXPOSED_EXPR/PAREN_EXPR
        if node.kind in [
            clang.cindex.CursorKind.UNEXPOSED_EXPR,
            clang.cindex.CursorKind.PAREN_EXPR,
        ]:
            children = list(node.get_children())
            if children:
                return unwrap_callee(children[0])
        return node

    for child in call_node.get_children():
        callee = unwrap_callee(child)
        if callee.kind in [
            clang.cindex.CursorKind.DECL_REF_EXPR,
            clang.cindex.CursorKind.MEMBER_REF_EXPR,
        ]:
            if callee.spelling:
                return callee.spelling
    return None

def is_callee_visible_from_file(callee_name, from_file):
    # Allow same-file calls
    callee_def = global_definitions.get(callee_name)
    if callee_def and callee_def.extent.start.file and callee_def.extent.start.file.name == from_file:
        return True

    included_headers = file_includes.get(from_file, set())
    included_basenames = file_include_basenames.get(from_file, set())

    # Check resolved include paths
    for header_path in included_headers:
        if callee_name in header_decl_map.get(header_path, set()):
            return True

    # Fallback: match by basename if include resolution failed
    for header_base in included_basenames:
        if callee_name in header_decl_basenames.get(header_base, set()):
            return True

    return False

def find_mpf_calls_from(func_node, visited=None):
    # Follow the call graph starting at func_node to find mpf_mfs_open
    if visited is None:
        visited = set()
    if not func_node or func_node.spelling in visited:
        return []
    visited.add(func_node.spelling)

    mpf_calls = []

    def walk(node):
        if node.kind == clang.cindex.CursorKind.CALL_EXPR:
            callee_name = get_called_function_name(node)
            # Debug: show call expression and resolved callee name
            print(f"[DEBUG] CALL_EXPR: {get_source_text(node)} -> {callee_name}")
            if callee_name == "mpf_mfs_open":
                mpf_calls.append(node)
                return
            if callee_name in global_definitions:
                from_file = func_node.extent.start.file.name if func_node.extent.start.file else None
                if from_file and is_callee_visible_from_file(callee_name, from_file):
                    mpf_calls.extend(find_mpf_calls_from(global_definitions[callee_name], visited))
                return
        for child in node.get_children():
            walk(child)

    walk(func_node)
    return mpf_calls

ast_json = {
    "translation_units": {
        file_path: cursor_to_dict(tu.cursor)
        for file_path, tu in file_tus.items()
    },
    "linkage": build_linkage_map(),
}
output_path = Path(__file__).parent / "output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ast_json, f, indent=2)
print(f"--- AST + linkage saved to {output_path} ---")

def build_callsite_id(call_node):
    callee = get_called_function_name(call_node) or "unknown"
    file_name = Path(call_node.location.file.name).name if call_node.location.file else "unknown"
    line = call_node.location.line if call_node.location else 0
    return f"{file_name}:{line}:{callee}"

print("--- Searching for mpf_mfs_open calls from main() ---")
main_func = global_definitions.get("main")
if not main_func:
    print("Error: main() function not found in the parsed files.")
    sys.exit(1)

all_calls = find_mpf_calls_from(main_func)
callsite_index = {build_callsite_id(call): call for call in all_calls}

if not all_calls:
    print("No calls to mpf_mfs_open found from main() call tree.")
    sys.exit(0)

print(f"\n--- Found {len(all_calls)} call(s)---")
for i, call in enumerate(all_calls):
    containing = find_containing_function(call)
    func_name  = containing.spelling if containing else "<failed to detect>"
    print(f"{i+1}: {Path(call.location.file.name).name}:{call.location.line}")
    print(f"  in function: {func_name}()")
    print(f"  -> {get_source_text(call)}\n")

try:
    selected_call = all_calls[0]
    containing_func = find_containing_function(selected_call)

    print(f"\n--- Tracing call--- ")
    print(f"Function: {containing_func.spelling}()")
    print(f"File: {Path(selected_call.location.file.name).name}:{selected_call.location.line}")
    print(f"Call: {get_source_text(selected_call)}")

    args = extract_arguments_from_call(selected_call)
    print(f"Arguments:{len(args)}")

    param_num = 3
    if not (1 <= param_num <= len(args)):
        print("Invalid parameter number.")
        sys.exit(1)

    result = deep_resolve(args[param_num - 1], selected_call, containing_func)
    print(f"\n--- Resolved Value---\n -> {result}")

    if should_trigger_llm(result):
        callsite_id = build_callsite_id(selected_call)
        orchestrator = LLMResolverOrchestrator()
        llm_result = orchestrator.resolve(result, callsite_id)
        llm_value = llm_result.get("value")
        if llm_value is not None:
            print(f"\n--- LLM Resolved Value---\n -> {llm_value}")
        else:
            reason = llm_result.get("reason", "unresolved")
            print(f"\n--- LLM Resolver---\n -> {reason}")
        # Debug: show macro resolution for APL_FILENO_DATA3
        macro_debug = get_macro_value("APL_FILENO_DATA3")
        if macro_debug:
            print(f"--- Macro Debug---\nAPL_FILENO_DATA3 = {macro_debug.get('value')}")
        else:
            print("--- Macro Debug---\nAPL_FILENO_DATA3 not found")

except Exception as e:
    print(f"Error during tracing: {e}")
