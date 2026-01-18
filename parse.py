import importlib
import json
from pathlib import Path
import re
import sys

# Load clang and its cindex module dynamically so linters don't flag missing imports
clang = importlib.import_module("clang")
importlib.import_module("clang.cindex")

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
ast_json = {
    file_path: cursor_to_dict(tu.cursor)
    for file_path, tu in file_tus.items()
}
output_path = Path(__file__).parent / "output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(ast_json, f, indent=2)
print(f"--- AST saved to {output_path} ---")

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

print("--- Searching for mpf_mfs_open calls from main() ---")
main_func = global_definitions.get("main")
if not main_func:
    print("Error: main() function not found in the parsed files.")
    sys.exit(1)

all_calls = find_mpf_calls_from(main_func)

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

except Exception as e:
    print(f"Error during tracing: {e}")

