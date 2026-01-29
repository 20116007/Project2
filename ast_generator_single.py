#!/usr/bin/env python3
"""
C Code AST and Call Graph Generator - Single File Version
Parses C code files, builds Abstract Syntax Trees (AST), and generates call graphs starting from main()
"""
import os
import sys
import re
import json
import argparse
from typing import Dict, List, Set, Optional
from collections import defaultdict

try:
    from clang import cindex  # type: ignore
    CLANG_AVAILABLE = True
except Exception:
    cindex = None  # type: ignore
    CLANG_AVAILABLE = False


# ============================================================================
# C File Parser
# ============================================================================

class CFileParser:
    """Parser for C source files"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.moove_header_path = os.path.join(project_root, "moove_header")
        self.src_analysis_path = os.path.join(project_root, "src_analysis")
        self.src_path = os.path.join(self.src_analysis_path, "src")
        self.include_path = os.path.join(self.src_analysis_path, "include")
        
    def list_apl_folders(self) -> List[str]:
        """List available apl folders under src_analysis/src (e.g., apl001, apl002)."""
        if not os.path.exists(self.src_path):
            return []

        apl_folders: List[str] = []
        for item in os.listdir(self.src_path):
            item_path = os.path.join(self.src_path, item)
            if os.path.isdir(item_path) and item.startswith("apl"):
                apl_folders.append(item)

        return sorted(apl_folders)

    def find_c_files_in_apl(self, apl_folder: str) -> List[str]:
        """
        Find all C files within a specific apl folder (e.g., apl001) under src_analysis/src.
        """
        c_files: List[str] = []
        if not os.path.exists(self.src_path):
            return c_files

        apl_folder = apl_folder.strip()
        apl_path = os.path.join(self.src_path, apl_folder)
        if not (apl_folder and os.path.isdir(apl_path)):
            return c_files

        for root, dirs, files in os.walk(apl_path):
            for file in files:
                if file.endswith((".c", ".C")):
                    c_files.append(os.path.join(root, file))

        return c_files

    def find_all_c_files(self) -> List[str]:
        """Find all C files in all apl folders (apl001, apl002, etc.) within src_analysis/src."""
        c_files: List[str] = []
        for apl in self.list_apl_folders():
            c_files.extend(self.find_c_files_in_apl(apl))
        return c_files
    
    def find_all_header_files(self) -> List[str]:
        """Find all header files in moove_header and include directories"""
        header_files = []
        
        # Search in moove_header
        if os.path.exists(self.moove_header_path):
            for root, dirs, files in os.walk(self.moove_header_path):
                for file in files:
                    if file.endswith(('.h', '.H')):
                        header_files.append(os.path.join(root, file))
        
        # Search in include
        if os.path.exists(self.include_path):
            for root, dirs, files in os.walk(self.include_path):
                for file in files:
                    if file.endswith(('.h', '.H')):
                        header_files.append(os.path.join(root, file))
        
        # Search in src subdirectories for local headers
        if os.path.exists(self.src_path):
            for root, dirs, files in os.walk(self.src_path):
                for file in files:
                    if file.endswith(('.h', '.H')):
                        header_files.append(os.path.join(root, file))
        
        return header_files
    
    def parse_c_file(self, file_path: str) -> Dict:
        """Parse a C file and extract functions, includes, and main function"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        result = {
            'file_path': file_path,
            'includes': self._extract_includes(content),
            'functions': self._extract_functions(content),
            'has_main': self._has_main_function(content),
            'main_function': None
        }
        
        if result['has_main']:
            result['main_function'] = self._extract_main_function(content)
        
        return result
    
    def _extract_includes(self, content: str) -> List[str]:
        """Extract #include directives"""
        includes = []
        # Match both #include <header.h> and #include "header.h"
        pattern = r'#include\s*[<"]([^>"]+)[>"]'
        matches = re.findall(pattern, content)
        return matches
    
    def _extract_functions(self, content: str) -> List[Dict]:
        """Extract function definitions from C code"""
        functions = []
        
        # Pattern to match function definitions
        # Pattern: return_type function_name(parameters) { ... }
        pattern = r'(\w+(?:\s+\*+)?)\s+(\w+)\s*\([^)]*\)\s*\{'
        
        matches = re.finditer(pattern, content, re.MULTILINE)
        
        for match in matches:
            return_type = match.group(1).strip()
            func_name = match.group(2).strip()
            
            # Skip if it's a variable declaration
            if func_name in ['if', 'while', 'for', 'switch', 'return']:
                continue
            
            # Extract parameters (simplified)
            func_start = match.end()
            param_match = re.search(r'\(([^)]*)\)', match.group(0))
            params = param_match.group(1) if param_match else ""
            
            # Find function body end
            brace_count = 1
            func_end = func_start
            i = func_start
            while i < len(content) and brace_count > 0:
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                i += 1
            func_end = i
            
            functions.append({
                'name': func_name,
                'return_type': return_type,
                'parameters': params,
                'start_pos': match.start(),
                'end_pos': func_end,
                'body': content[match.start():func_end]
            })
        
        return functions
    
    def _has_main_function(self, content: str) -> bool:
        """Check if file contains main function"""
        # Match various main function signatures
        patterns = [
            r'int\s+main\s*\(',
            r'void\s+main\s*\(',
            r'int\s+main\s*\([^)]*\)',
            r'void\s+main\s*\([^)]*\)'
        ]
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _extract_main_function(self, content: str) -> Dict:
        """Extract main function details"""
        # Find main function
        pattern = r'(int|void)\s+main\s*\(([^)]*)\)\s*\{'
        match = re.search(pattern, content)
        
        if match:
            return_type = match.group(1)
            params = match.group(2) if match.group(2) else ""
            
            # Find function body
            func_start = match.start()
            brace_count = 1
            i = match.end()
            while i < len(content) and brace_count > 0:
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                i += 1
            
            return {
                'name': 'main',
                'return_type': return_type,
                'parameters': params,
                'start_pos': func_start,
                'end_pos': i,
                'body': content[func_start:i]
            }
        return None
    
    def parse_header_file(self, file_path: str) -> Dict:
        """Parse a header file and extract function declarations"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return {
            'file_path': file_path,
            'function_declarations': self._extract_function_declarations(content),
            'includes': self._extract_includes(content)
        }
    
    def _extract_function_declarations(self, content: str) -> List[Dict]:
        """Extract function declarations from header files"""
        declarations = []
        
        # Pattern for function declarations (ending with ;)
        # return_type function_name(parameters);
        pattern = r'(\w+(?:\s+\*+)?)\s+(\w+)\s*\([^)]*\)\s*;'
        
        matches = re.finditer(pattern, content, re.MULTILINE)
        
        for match in matches:
            return_type = match.group(1).strip()
            func_name = match.group(2).strip()
            
            # Skip common keywords
            if func_name in ['if', 'while', 'for', 'switch', 'return', 'typedef']:
                continue
            
            param_match = re.search(r'\(([^)]*)\)', match.group(0))
            params = param_match.group(1) if param_match else ""
            
            declarations.append({
                'name': func_name,
                'return_type': return_type,
                'parameters': params
            })
        
        return declarations
    
    def find_header_file(self, header_name: str) -> Optional[str]:
        """Find a header file by name in all header directories"""
        # Try exact match first
        search_paths = [
            self.moove_header_path,
            self.include_path,
            self.src_path
        ]
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file == header_name or file == os.path.basename(header_name):
                        return os.path.join(root, file)
        
        return None


# ============================================================================
# AST Builder
# ============================================================================

class ASTBuilder:
    """Builds AST for C code with function validation"""
    
    def __init__(self, parser: CFileParser):
        self.parser = parser
        self.header_declarations: Dict[str, List[Dict]] = {}  # func_name -> [declarations]
        self.function_definitions: Dict[str, Dict] = {}  # func_name -> definition
        self.validated_functions: Set[str] = set()
        self.missing_headers: List[str] = []
        self.functions_not_in_headers: List[str] = []
        
    def build_ast(self, c_files_data: List[Dict]) -> Dict:
        """Build AST starting from main function"""
        # Step 1: Parse all header files and collect declarations
        self._parse_all_headers()
        
        # Step 2: Find the file with main function
        main_file_data = None
        for file_data in c_files_data:
            if file_data.get('has_main'):
                main_file_data = file_data
                break
        
        if not main_file_data:
            return {
                'error': 'No main function found in any C file',
                'ast': None
            }
        
        # Step 3: Validate includes in all C files
        for file_data in c_files_data:
            self._validate_includes(file_data['includes'])
        
        # Step 4: Process all C files and collect function definitions
        for file_data in c_files_data:
            self._process_c_file(file_data)
        
        # Step 5: Validate functions - only include if declared in header and defined in C file
        self._validate_functions()
        
        # Step 6: Build AST starting from main
        ast = self._build_ast_from_main(main_file_data)
        
        return {
            'ast': ast,
            'main_file': main_file_data['file_path'],
            'validated_functions': list(self.validated_functions),
            'missing_headers': self.missing_headers,
            'functions_not_in_headers': self.functions_not_in_headers,
            'function_definitions': {k: {
                'name': v['name'],
                'return_type': v['return_type'],
                'file': v.get('file_path', 'unknown')
            } for k, v in self.function_definitions.items()}
        }
    
    def _parse_all_headers(self):
        """Parse all header files and collect function declarations"""
        header_files = self.parser.find_all_header_files()
        
        for header_path in header_files:
            try:
                header_data = self.parser.parse_header_file(header_path)
                for decl in header_data['function_declarations']:
                    func_name = decl['name']
                    if func_name not in self.header_declarations:
                        self.header_declarations[func_name] = []
                    self.header_declarations[func_name].append({
                        **decl,
                        'header_file': header_path
                    })
            except Exception as e:
                print(f"Error parsing header {header_path}: {e}")
    
    def _validate_includes(self, includes: List[str]):
        """Validate that all included headers exist"""
        for include in includes:
            header_path = self.parser.find_header_file(include)
            if not header_path:
                self.missing_headers.append(include)
    
    def _process_c_file(self, file_data: Dict):
        """Process a C file and collect function definitions"""
        for func in file_data['functions']:
            func_name = func['name']
            self.function_definitions[func_name] = {
                **func,
                'file_path': file_data['file_path']
            }
    
    def _validate_functions(self):
        """Validate that functions are declared in headers and defined in C files"""
        for func_name, definition in self.function_definitions.items():
            # Skip main function
            if func_name == 'main':
                self.validated_functions.add(func_name)
                continue
            
            # Check if function is declared in any header
            if func_name in self.header_declarations:
                # Function is declared in header and defined in C file - valid
                self.validated_functions.add(func_name)
            else:
                # Function not found in any header
                self.functions_not_in_headers.append(func_name)
    
    def _build_ast_from_main(self, main_file_data: Dict) -> Dict:
        """Build AST starting from main function"""
        main_func = main_file_data['main_function']
        if not main_func:
            return None
        
        # Extract function calls from main
        function_calls = self._extract_function_calls(main_func['body'])
        
        ast = {
            'type': 'function',
            'name': 'main',
            'return_type': main_func['return_type'],
            'parameters': main_func['parameters'],
            'file': main_file_data['file_path'],
            'calls': []
        }
        
        # Build AST for called functions
        for call in function_calls:
            if call in self.validated_functions:
                call_ast = self._build_function_ast(call, set())  # visited set to avoid cycles
                if call_ast:
                    ast['calls'].append(call_ast)
        
        return ast
    
    def _extract_function_calls(self, code: str) -> List[str]:
        """Extract function calls from code"""
        calls = []
        # Pattern: function_name(...) - more precise
        # Skip function pointers and complex expressions for now
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.finditer(pattern, code)
        
        # C keywords and common library functions to skip
        keywords = {
            'if', 'while', 'for', 'switch', 'return', 'sizeof', 
            'printf', 'scanf', 'fprintf', 'fscanf', 'sprintf', 'sscanf',
            'malloc', 'free', 'calloc', 'realloc',
            'memset', 'memcpy', 'memmove', 'memcmp',
            'strlen', 'strcpy', 'strncpy', 'strcmp', 'strncmp', 'strcat', 'strncat',
            'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell',
            'exit', 'abort', 'assert', 'getchar', 'putchar', 'gets', 'puts'
        }
        
        for match in matches:
            func_name = match.group(1)
            # Skip keywords and already added functions
            if func_name not in keywords and func_name not in calls:
                # Additional check: make sure it's not part of a type declaration
                pos = match.start()
                if pos > 0:
                    # Check if preceded by a type keyword (simple heuristic)
                    before = code[max(0, pos-20):pos].strip()
                    if not re.search(r'\b(int|char|float|double|void|struct|union|enum|typedef)\s+$', before):
                        calls.append(func_name)
                else:
                    calls.append(func_name)
        
        return calls
    
    def _build_function_ast(self, func_name: str, visited: Set[str]) -> Optional[Dict]:
        """Build AST for a function"""
        if func_name in visited:
            return None  # Avoid cycles
        
        if func_name not in self.validated_functions:
            return None
        
        if func_name not in self.function_definitions:
            return None
        
        visited.add(func_name)
        definition = self.function_definitions[func_name]
        
        # Extract function calls from this function
        function_calls = self._extract_function_calls(definition['body'])
        
        ast = {
            'type': 'function',
            'name': func_name,
            'return_type': definition['return_type'],
            'parameters': definition['parameters'],
            'file': definition['file_path'],
            'calls': []
        }
        
        # Recursively build AST for called functions
        for call in function_calls:
            if call in self.validated_functions and call != func_name:
                call_ast = self._build_function_ast(call, visited.copy())
                if call_ast:
                    ast['calls'].append(call_ast)
        
        return ast


# ============================================================================
# Call Graph Generator
# ============================================================================

class CallGraphGenerator:
    """Generates call graph from AST"""
    
    def __init__(self):
        self.graph_nodes: List[Dict] = []
        self.graph_edges: List[Dict] = []
        self.visited: Set[str] = set()
    
    def build_call_graph(self, ast: Dict) -> Dict:
        """Build call graph from AST"""
        if not ast:
            return {'nodes': [], 'edges': []}
        
        self.graph_nodes = []
        self.graph_edges = []
        self.visited = set()
        
        self._traverse_ast(ast, None)
        
        return {
            'nodes': self.graph_nodes,
            'edges': self.graph_edges
        }
    
    def _traverse_ast(self, node: Dict, parent: Optional[str]):
        """Traverse AST and build graph nodes and edges"""
        if not node or node.get('type') != 'function':
            return
        
        func_name = node.get('name')
        if not func_name:
            return
        
        # Add node if not already added
        if func_name not in self.visited:
            self.graph_nodes.append({
                'id': func_name,
                'label': func_name,
                'return_type': node.get('return_type', 'void'),
                'file': node.get('file', 'unknown')
            })
            self.visited.add(func_name)
        
        # Add edge from parent to current function
        if parent:
            edge = {
                'from': parent,
                'to': func_name
            }
            # Avoid duplicate edges
            if edge not in self.graph_edges:
                self.graph_edges.append(edge)
        
        # Traverse called functions
        for call in node.get('calls', []):
            self._traverse_ast(call, func_name)
    
    def generate_dot_format(self, graph: Dict) -> str:
        """Generate Graphviz DOT format string"""
        dot = "digraph CallGraph {\n"
        dot += "  rankdir=TB;\n"
        dot += "  node [shape=box, style=rounded];\n\n"
        
        # Add nodes
        for node in graph['nodes']:
            label = f"{node['label']}\\n({node['return_type']})"
            dot += f'  "{node["id"]}" [label="{label}"];\n'
        
        dot += "\n"
        
        # Add edges
        for edge in graph['edges']:
            dot += f'  "{edge["from"]}" -> "{edge["to"]}";\n'
        
        dot += "}\n"
        return dot
    
    def generate_json(self, graph: Dict) -> str:
        """Generate JSON representation of call graph"""
        return json.dumps(graph, indent=2)
    
    def print_call_graph(self, graph: Dict):
        """Print call graph in text format"""
        print("\n=== Call Graph ===")
        print(f"Total nodes: {len(graph['nodes'])}")
        print(f"Total edges: {len(graph['edges'])}\n")
        
        # Build adjacency list
        adj_list = {}
        for edge in graph['edges']:
            if edge['from'] not in adj_list:
                adj_list[edge['from']] = []
            adj_list[edge['from']].append(edge['to'])
        
        # Print graph
        for node in graph['nodes']:
            func_name = node['id']
            print(f"{func_name} ({node['return_type']})")
            if func_name in adj_list:
                for called_func in adj_list[func_name]:
                    print(f"  -> {called_func}")
            print()


# ============================================================================
# Clang-based Analyzer (libclang)
# ============================================================================

def _clang_location_to_dict(loc) -> Optional[Dict]:
    try:
        if not loc or not loc.file:
            return None
        return {
            "file": str(loc.file.name),
            "line": int(loc.line),
            "column": int(loc.column),
        }
    except Exception:
        return None


def _clang_cursor_to_node(cur) -> Dict:
    """Convert a clang cursor into a JSON-serializable node."""
    try:
        kind = cur.kind.name
    except Exception:
        kind = "UNKNOWN"

    node = {
        "kind": kind,
        "spelling": getattr(cur, "spelling", "") or "",
        "displayname": getattr(cur, "displayname", "") or "",
        "usr": "",
        "location": _clang_location_to_dict(getattr(cur, "location", None)),
        "type": str(getattr(getattr(cur, "type", None), "spelling", "") or ""),
        "result_type": "",
    }

    try:
        node["usr"] = cur.get_usr()
    except Exception:
        node["usr"] = ""

    try:
        node["result_type"] = str(cur.result_type.spelling) if getattr(cur, "result_type", None) else ""
    except Exception:
        node["result_type"] = ""

    return node


class ClangAnalyzer:
    """
    Uses libclang to parse C/headers and extract:
    - function definitions (with clang cursors)
    - function declarations from headers
    - inclusion directives
    - call expressions per function
    """

    def __init__(
        self,
        include_dirs: List[str],
        clang_library_file: Optional[str] = None,
        clang_library_path: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ):
        if not CLANG_AVAILABLE:
            raise RuntimeError("clang python bindings not available. Install `clang` and libclang.")

        # Optional explicit libclang configuration (useful on macOS)
        if clang_library_file:
            cindex.Config.set_library_file(clang_library_file)
        elif clang_library_path:
            cindex.Config.set_library_path(clang_library_path)

        self.include_dirs = [d for d in include_dirs if d and os.path.isdir(d)]
        self.args = [
            "-x",
            "c",
            "-std=c99",
            "-ferror-limit=0",
        ]
        for d in self.include_dirs:
            self.args.append(f"-I{d}")

        if extra_args:
            self.args.extend(extra_args)

        self.index = cindex.Index.create()

    def _parse(self, path: str, is_header: bool = False):
        parse_args = list(self.args)
        if is_header:
            # Make clang treat file as header if needed
            parse_args = ["-x", "c-header"] + [a for a in parse_args if a not in ("-x", "c")]

        options = (
            cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            | cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES  # we re-parse for bodies via cursors; keep speed
        )
        # Note: We still can traverse calls if bodies are skipped => calls won't exist.
        # We'll override options when we need bodies.
        return self.index.parse(path, args=parse_args, options=options)

    def parse_source_with_bodies(self, path: str):
        parse_args = list(self.args)
        preamble_opt = getattr(cindex.TranslationUnit, "PARSE_PRECOMPILED_PREAMBLE", 0)
        options = cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD | preamble_opt
        return self.index.parse(path, args=parse_args, options=options)

    def iter_children(self, cursor):
        for c in cursor.get_children():
            yield c

    def collect_includes(self, tu) -> List[Dict]:
        includes: List[Dict] = []
        for cur in tu.cursor.get_children():
            try:
                if cur.kind == cindex.CursorKind.INCLUSION_DIRECTIVE:
                    inc = {
                        "spelling": cur.spelling,
                        "location": _clang_location_to_dict(cur.location),
                        "included_file": None,
                    }
                    try:
                        f = cur.get_included_file()
                        inc["included_file"] = str(f.name) if f else None
                    except Exception:
                        inc["included_file"] = None
                    includes.append(inc)
            except Exception:
                continue
        return includes

    def collect_function_definitions(self, tu, source_path: str) -> Dict[str, Dict]:
        defs: Dict[str, Dict] = {}

        def visit(cur):
            try:
                if cur.kind == cindex.CursorKind.FUNCTION_DECL and cur.is_definition():
                    loc = cur.location
                    if loc and loc.file and os.path.abspath(str(loc.file.name)) == os.path.abspath(source_path):
                        name = cur.spelling
                        defs[name] = {
                            "cursor": cur,
                            "name": name,
                            "return_type": str(cur.result_type.spelling) if cur.result_type else "",
                            "parameters": [str(p.type.spelling) for p in cur.get_arguments()],
                            "file_path": source_path,
                            "usr": cur.get_usr() if hasattr(cur, "get_usr") else "",
                            "location": _clang_location_to_dict(cur.location),
                        }
            except Exception:
                pass

            for ch in cur.get_children():
                visit(ch)

        visit(tu.cursor)
        return defs

    def collect_function_declarations_from_header(self, header_path: str) -> Set[str]:
        """
        Parse a header and return declared function names.
        We parse as a standalone TU; for best results ensure include dirs include dependencies.
        """
        try:
            tu = self._parse(header_path, is_header=True)
        except Exception:
            # Fallback: try parsing as C (some headers aren't pure)
            tu = self._parse(header_path, is_header=False)

        names: Set[str] = set()

        def visit(cur):
            try:
                if cur.kind == cindex.CursorKind.FUNCTION_DECL and not cur.is_definition():
                    loc = cur.location
                    if loc and loc.file and os.path.abspath(str(loc.file.name)) == os.path.abspath(header_path):
                        if cur.spelling:
                            names.add(cur.spelling)
            except Exception:
                pass
            for ch in cur.get_children():
                visit(ch)

        visit(tu.cursor)
        return names

    def collect_calls_in_function(self, func_cursor) -> List[Dict]:
        calls: List[Dict] = []

        def visit(cur):
            try:
                if cur.kind == cindex.CursorKind.CALL_EXPR:
                    callee_name = ""
                    try:
                        ref = cur.referenced
                        if ref and ref.spelling:
                            callee_name = ref.spelling
                    except Exception:
                        callee_name = ""

                    if not callee_name:
                        # Try to infer from children (e.g., DeclRefExpr)
                        for ch in cur.get_children():
                            try:
                                if ch.spelling:
                                    callee_name = ch.spelling
                                    break
                            except Exception:
                                continue

                    calls.append({
                        "name": callee_name or "<unknown>",
                        "cursor": cur,
                        "location": _clang_location_to_dict(cur.location),
                    })
            except Exception:
                pass

            for ch in cur.get_children():
                visit(ch)

        visit(func_cursor)
        # Deduplicate by name while preserving order
        seen: Set[str] = set()
        out: List[Dict] = []
        for c in calls:
            n = c["name"]
            if n not in seen:
                seen.add(n)
                out.append(c)
        return out


class ClangASTBuilder:
    """Builds an AST + call graph starting from main() using clang nodes."""

    def __init__(self, header_declared: Set[str], func_defs: Dict[str, Dict], call_map: Dict[str, List[str]]):
        self.header_declared = header_declared
        self.func_defs = func_defs
        self.call_map = call_map

    def build_from_main(self) -> Dict:
        if "main" not in self.func_defs:
            return {"error": "No main() definition found in parsed C files", "ast": None}

        visited: Set[str] = set()

        def build_func(name: str) -> Optional[Dict]:
            if name in visited:
                return None
            if name != "main":
                # Only include if declared in some header AND defined in parsed C files
                if name not in self.header_declared:
                    return None
                if name not in self.func_defs:
                    return None

            if name not in self.func_defs:
                return None

            visited.add(name)
            info = self.func_defs[name]
            node = {
                "type": "function",
                "name": name,
                "return_type": info.get("return_type", ""),
                "parameters": info.get("parameters", []),
                "file": info.get("file_path", "unknown"),
                "clang_node": _clang_cursor_to_node(info.get("cursor")),
                "calls": [],
            }

            for callee in self.call_map.get(name, []):
                child = build_func(callee)
                if child:
                    node["calls"].append(child)
            return node

        ast = build_func("main")
        return {"ast": ast, "main_file": self.func_defs["main"].get("file_path", "unknown")}


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate AST and Call Graph for C code')
    parser.add_argument('--project-root', type=str, required=True,
                       help='Path to project root directory')
    parser.add_argument('--apl', type=str, default=None,
                       help='APL folder name under src_analysis/src (e.g., apl001). If omitted, prompts.')
    parser.add_argument('--clang-lib', type=str, default=None,
                       help='Path to libclang dynamic library file (optional).')
    parser.add_argument('--clang-lib-path', type=str, default=None,
                       help='Path to directory containing libclang (optional).')
    parser.add_argument('--clang-arg', action='append', default=[],
                       help='Extra clang argument (repeatable), e.g. --clang-arg=-DDEBUG')
    parser.add_argument('--output-ast', type=str, default='ast_output.json',
                       help='Output file for AST (JSON)')
    parser.add_argument('--output-graph', type=str, default='call_graph.json',
                       help='Output file for call graph (JSON)')
    parser.add_argument('--output-dot', type=str, default='call_graph.dot',
                       help='Output file for call graph (DOT format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    project_root = os.path.abspath(args.project_root)
    
    if not os.path.exists(project_root):
        print(f"Error: Project root directory does not exist: {project_root}")
        sys.exit(1)
    
    print(f"Project root: {project_root}")
    print("=" * 60)
    
    # Initialize parser
    c_parser = CFileParser(project_root)
    
    # Pick apl folder (prompt if not provided)
    apl = args.apl.strip() if isinstance(args.apl, str) and args.apl else ""
    if not apl:
        available_apl = c_parser.list_apl_folders()
        print("\nAvailable apl folders:")
        if available_apl:
            for name in available_apl:
                print(f"  - {name}")
        else:
            print("  (none found under src_analysis/src)")

        apl = input("\nEnter apl folder name (e.g., apl001): ").strip()

    # Step 1: Find C files in the selected apl folder
    print(f"\n[Step 1] Finding C files in {apl}...")
    c_files = c_parser.find_c_files_in_apl(apl)
    print(f"Found {len(c_files)} C file(s) in {apl}")
    if args.verbose:
        for cf in c_files:
            print(f"  - {cf}")
    
    if not c_files:
        print(f"Error: No C files found in apl folder: {apl}")
        sys.exit(1)
    
    if not CLANG_AVAILABLE:
        print("\nError: clang python bindings not installed.")
        print("Install with: uv pip install clang  (and install libclang on your system)")
        print("macOS (Homebrew): brew install llvm")
        sys.exit(1)

    # Build include directories for clang
    apl_path = os.path.join(c_parser.src_path, apl)
    include_dirs: List[str] = [
        c_parser.include_path,
        apl_path,
        c_parser.src_path,
        project_root,
    ]
    if os.path.isdir(c_parser.moove_header_path):
        include_dirs.append(c_parser.moove_header_path)
        for root, dirs, _files in os.walk(c_parser.moove_header_path):
            for d in dirs:
                include_dirs.append(os.path.join(root, d))

    # Step 2: Parse all C files with clang (real AST nodes)
    print("\n[Step 2] Parsing C files with clang...")
    analyzer = ClangAnalyzer(
        include_dirs=include_dirs,
        clang_library_file=args.clang_lib,
        clang_library_path=args.clang_lib_path,
        extra_args=args.clang_arg,
    )

    all_func_defs: Dict[str, Dict] = {}
    call_map: Dict[str, List[str]] = {}
    missing_headers_set: Set[str] = set()
    main_file_found = False

    for c_file in c_files:
        try:
            tu = analyzer.parse_source_with_bodies(c_file)

            includes = analyzer.collect_includes(tu)
            for inc in includes:
                # If clang couldn't resolve it, treat as missing (typically project headers)
                if inc.get("included_file") is None:
                    name = (inc.get("spelling") or "").strip()
                    if name:
                        missing_headers_set.add(name)

            defs = analyzer.collect_function_definitions(tu, c_file)
            if "main" in defs:
                main_file_found = True
                print(f"  Found main() in: {c_file}")

            # Merge function definitions (later files overwrite on name conflict)
            for name, info in defs.items():
                all_func_defs[name] = info

            # Extract calls for each function definition
            for name, info in defs.items():
                calls = analyzer.collect_calls_in_function(info["cursor"])
                call_map[name] = [c["name"] for c in calls if c.get("name") and c["name"] != "<unknown>"]

            if args.verbose:
                print(f"  Parsed: {c_file}")
                print(f"    Functions: {sorted(list(defs.keys()))}")
        except Exception as e:
            print(f"  Error parsing with clang {c_file}: {e}")

    if not main_file_found:
        print("Warning: No main() function found in any parsed C file")

    # Step 3: Parse header declarations with clang
    print("\n[Step 3] Finding and parsing header files (declarations) with clang...")
    header_files = c_parser.find_all_header_files()
    print(f"Found {len(header_files)} header file(s)")
    if args.verbose:
        for hf in header_files[:10]:
            print(f"  - {hf}")
        if len(header_files) > 10:
            print(f"  ... and {len(header_files) - 10} more")

    header_declared: Set[str] = set()
    for hf in header_files:
        try:
            header_declared |= analyzer.collect_function_declarations_from_header(hf)
        except Exception:
            # If a header doesn't parse standalone, skip it (still declared via other headers)
            continue

    # Validate functions vs headers (your rule)
    functions_not_in_headers = sorted([n for n in all_func_defs.keys() if n != "main" and n not in header_declared])
    validated_functions = sorted(["main"] + [n for n in all_func_defs.keys() if n != "main" and n in header_declared])

    # Step 4: Build AST from clang nodes (starting at main)
    print("\n[Step 4] Building AST from clang nodes...")
    clang_builder = ClangASTBuilder(header_declared=header_declared, func_defs=all_func_defs, call_map=call_map)
    built = clang_builder.build_from_main()
    if built.get("error"):
        print(f"Error: {built['error']}")
        sys.exit(1)

    ast_result = {
        "ast": built["ast"],
        "main_file": built["main_file"],
        "validated_functions": validated_functions,
        "missing_headers": sorted(list(missing_headers_set)),
        "functions_not_in_headers": functions_not_in_headers,
        "function_definitions": {
            k: {
                "name": v.get("name", k),
                "return_type": v.get("return_type", ""),
                "file": v.get("file_path", "unknown"),
                "usr": v.get("usr", ""),
                "location": v.get("location", None),
            }
            for k, v in all_func_defs.items()
        },
    }

    print(f"Main file: {ast_result['main_file']}")
    print(f"Validated functions: {len(ast_result['validated_functions'])}")
    
    if ast_result['missing_headers']:
        print(f"\nMissing headers ({len(ast_result['missing_headers'])}):")
        for header in ast_result['missing_headers']:
            print(f"  - {header} (not present in any header folder)")
    
    if ast_result['functions_not_in_headers']:
        print(f"\nFunctions not in headers ({len(ast_result['functions_not_in_headers'])}):")
        for func in ast_result['functions_not_in_headers']:
            print(f"  - {func} (not present in any header file)")
    
    # Step 5: Generate call graph
    print("\n[Step 5] Generating call graph...")
    call_graph_gen = CallGraphGenerator()
    call_graph = call_graph_gen.build_call_graph(ast_result['ast'])
    
    print(f"Call graph nodes: {len(call_graph['nodes'])}")
    print(f"Call graph edges: {len(call_graph['edges'])}")
    
    # Print call graph
    call_graph_gen.print_call_graph(call_graph)
    
    # Step 6: Save outputs
    print("\n[Step 6] Saving outputs...")
    
    # Save AST
    with open(args.output_ast, 'w') as f:
        json.dump({
            'ast': ast_result['ast'],
            'metadata': {
                'apl_folder': apl,
                'main_file': ast_result['main_file'],
                'validated_functions': ast_result['validated_functions'],
                'missing_headers': ast_result['missing_headers'],
                'functions_not_in_headers': ast_result['functions_not_in_headers'],
                'function_definitions': ast_result['function_definitions']
            }
        }, f, indent=2)
    print(f"  AST saved to: {args.output_ast}")
    
    # Save call graph JSON
    with open(args.output_graph, 'w') as f:
        json.dump(call_graph, f, indent=2)
    print(f"  Call graph (JSON) saved to: {args.output_graph}")
    
    # Save call graph DOT
    dot_content = call_graph_gen.generate_dot_format(call_graph)
    with open(args.output_dot, 'w') as f:
        f.write(dot_content)
    print(f"  Call graph (DOT) saved to: {args.output_dot}")
    
    print("\n" + "=" * 60)
    print("AST and Call Graph generation completed!")
    print("\nTo visualize the call graph, run:")
    print(f"  dot -Tpng {args.output_dot} -o call_graph.png")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
