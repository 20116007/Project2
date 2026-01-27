#!/usr/bin/env python3
"""
AST Generator for C Code using libclang

This script parses C code files and generates an Abstract Syntax Tree (AST)
with proper linkages between function calls and their declarations/definitions.
It handles complex folder structures with local headers, global headers, and middleware headers.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from clang.cindex import Index, TranslationUnit, Cursor, CursorKind, SourceLocation
from clang.cindex import Config

# Configure libclang path (adjust if needed)
# Config.set_library_path('/usr/lib/llvm-*/lib')  # Uncomment and adjust if needed

class ASTNode:
    """Represents a node in the AST"""
    def __init__(self, name: str, kind: str, location: str, node_type: str = ""):
        self.name = name
        self.kind = kind  # 'function', 'variable', 'macro', 'struct', 'enum', etc.
        self.location = location  # File path and line number
        self.node_type = node_type  # Return type for functions, type for variables
        self.children: List['ASTNode'] = []
        self.parent: Optional['ASTNode'] = None
        self.linked_to: List['ASTNode'] = []  # Links to declarations/definitions
        self.is_declaration = False
        self.is_definition = False
        self.parameters: List[str] = []  # For functions
        self.is_macro = False
        
    def __repr__(self):
        return f"{self.kind}:{self.name}@{self.location}"


class ASTGenerator:
    """Main class for generating AST from C code"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.index = Index.create()
        
        # Storage for AST nodes
        self.functions: Dict[str, List[ASTNode]] = defaultdict(list)  # name -> list of nodes
        self.variables: Dict[str, List[ASTNode]] = defaultdict(list)
        self.macros: Dict[str, List[ASTNode]] = defaultdict(list)
        self.structs: Dict[str, List[ASTNode]] = defaultdict(list)
        self.enums: Dict[str, List[ASTNode]] = defaultdict(list)
        
        # Track which files have been parsed
        self.parsed_files: Set[str] = set()
        
        # Include paths
        self.include_paths: List[str] = []
        
        # Track function call linkages
        self.function_calls: List[Tuple[ASTNode, str, str]] = []  # (caller_node, function_name, location)
        
        # Track function calls within each function (for call graph)
        self.function_call_map: Dict[ASTNode, List[Tuple[ASTNode, str, str]]] = defaultdict(list)
        # Maps: function_node -> list of (call_node, function_name, location)
        
        # Track current function context during parsing
        self.current_function: Optional[ASTNode] = None
        
    def find_all_subdirectories(self, directory: Path) -> List[Path]:
        """
        Recursively find all subdirectories in a given directory.
        This is used for global headers and middleware headers that have nested folder structures.
        For example: MIDDLE_HEADER/FILE/subfolder/header.h or GLOBAL_HEADER/subfolder/header.h
        """
        subdirs = []
        if not directory.exists() or not directory.is_dir():
            return subdirs
        
        # Add the directory itself
        subdirs.append(directory)
        
        # Recursively find all subdirectories
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    subdirs.append(item)
                    # Recursively get subdirectories of this subdirectory
                    subdirs.extend(self.find_all_subdirectories(item))
        except PermissionError:
            pass  # Skip directories we can't access
        
        return subdirs
    
    def setup_include_paths(self, folder_path: Path):
        """
        Setup include paths for the parser based on the directory structure:
        
        Expected structure:
        - goal/src_analysis/src/apl001/, apl002/, etc. (source folders)
        - goal/src_analysis/include/ (global headers with nested folders)
        - goal/src_analysis/libapl/ (libapl implementation files)
        - goal/moove_header/ (middleware headers with nested folders)
        
        1. The folder itself (where C files are) - for headers in the same directory
        2. Local header folder in the current folder
        3. Global header folder (src_analysis/include with all nested subdirectories)
        4. Middleware headers folder (moove_header with all nested subdirectories)
        5. libapl folder (src_analysis/libapl)
        6. System include paths
        """
        self.include_paths = []
        
        # 0. Add the folder itself - headers can be in the same directory as C files
        # This allows includes like #include "local_header.h" when header is in same folder
        self.include_paths.append(str(folder_path))
        print(f"  Added folder path: {folder_path}")
        
        # 1. Local header folder in the current folder
        local_header = folder_path / "header"
        if local_header.exists():
            self.include_paths.append(str(local_header))
            print(f"  Added local header folder: {local_header}")
        
        # 2. Global header folder - check for src_analysis/include structure first
        # Check if we're in the src_analysis/src/ structure
        src_analysis_include = None
        if "src_analysis" in str(folder_path):
            # Try to find src_analysis/include relative to current folder
            # folder_path might be: .../src_analysis/src/apl001/
            # We need: .../src_analysis/include/
            path_parts = folder_path.parts
            if "src_analysis" in path_parts:
                # Find src_analysis index and build path to include
                src_idx = path_parts.index("src_analysis")
                src_analysis_base = Path(*path_parts[:src_idx + 1])
                src_analysis_include = src_analysis_base / "include"
        
        # Also check at base_path level
        src_analysis_base_path = self.base_path / "src_analysis" / "include"
        if src_analysis_base_path.exists():
            src_analysis_include = src_analysis_base_path
        
        global_folders_found = []
        
        # Add src_analysis/include if found
        if src_analysis_include and src_analysis_include.exists():
            global_subdirs = self.find_all_subdirectories(src_analysis_include)
            for subdir in global_subdirs:
                if str(subdir) not in self.include_paths:
                    self.include_paths.append(str(subdir))
            print(f"  Found src_analysis/include with {len(global_subdirs)} directories (including nested)")
            global_folders_found.append(src_analysis_include)
        
        # Fallback: check for other common global header folder names
        for name in ["global_headers", "global_header", "headers", "include", "GLOBAL_HEADER"]:
            global_path = self.base_path / name
            if global_path.exists() and global_path not in global_folders_found:
                global_subdirs = self.find_all_subdirectories(global_path)
                for subdir in global_subdirs:
                    if str(subdir) not in self.include_paths:
                        self.include_paths.append(str(subdir))
                print(f"  Found {name} with {len(global_subdirs)} directories (including nested)")
                global_folders_found.append(global_path)
        
        # 3. Middleware headers folder - recursively add all subdirectories
        # First check for moove_header (exact name as mentioned by user)
        middleware_headers = self.base_path / "moove_header"
        if middleware_headers.exists():
            # Get all subdirectories recursively (including the parent folder itself)
            mw_subdirs = self.find_all_subdirectories(middleware_headers)
            for subdir in mw_subdirs:
                if str(subdir) not in self.include_paths:  # Avoid duplicates
                    self.include_paths.append(str(subdir))
            print(f"  Found moove_header with {len(mw_subdirs)} directories (including nested)")
        
        # Also check for other common middleware folder names (including old MIDDLE_HEADER for backward compatibility)
        for name in ["MIDDLE_HEADER", "middleware_headers", "middleware", "middleware_include", "mw_headers"]:
            mw_path = self.base_path / name
            if mw_path.exists() and mw_path != middleware_headers:
                # Get all subdirectories recursively
                mw_subdirs = self.find_all_subdirectories(mw_path)
                for subdir in mw_subdirs:
                    if str(subdir) not in self.include_paths:  # Avoid duplicates
                        self.include_paths.append(str(subdir))
                print(f"  Found {name} with {len(mw_subdirs)} directories (including nested)")
        
        # 4. libapl folder - check for src_analysis/libapl structure first
        libapl_path = None
        if "src_analysis" in str(folder_path):
            # Try to find src_analysis/libapl relative to current folder
            path_parts = folder_path.parts
            if "src_analysis" in path_parts:
                src_idx = path_parts.index("src_analysis")
                src_analysis_base = Path(*path_parts[:src_idx + 1])
                libapl_path = src_analysis_base / "libapl"
        
        # Also check at base_path level
        libapl_base_path = self.base_path / "src_analysis" / "libapl"
        if libapl_base_path.exists():
            libapl_path = libapl_base_path
        
        # Add src_analysis/libapl if found
        if libapl_path and libapl_path.exists():
            if str(libapl_path) not in self.include_paths:
                self.include_paths.append(str(libapl_path))
                print(f"  Added src_analysis/libapl folder: {libapl_path}")
        
        # Fallback: check for libapl in current folder
        local_libapl = folder_path / "libapl"
        if local_libapl.exists():
            if str(local_libapl) not in self.include_paths:
                self.include_paths.append(str(local_libapl))
                print(f"  Added local libapl folder: {local_libapl}")
        
        # Fallback: check if libapl is at base level
        libapl_base = self.base_path / "libapl"
        if libapl_base.exists():
            if str(libapl_base) not in self.include_paths:
                self.include_paths.append(str(libapl_base))
                print(f"  Added base libapl folder: {libapl_base}")
        
        # 5. System include paths (for standard C headers)
        self.include_paths.extend([
            "/usr/include",
            "/usr/local/include",
            "/usr/include/clang",
        ])
        
        print(f"\nTotal include paths: {len(self.include_paths)}")
        print(f"Sample include paths: {self.include_paths[:10]}...")  # Show first 10 to avoid clutter
    
    def get_include_args(self) -> List[str]:
        """Convert include paths to compiler arguments"""
        args = []
        for path in self.include_paths:
            args.extend(["-I", path])
        return args
    
    def find_c_files(self, folder_path: Path) -> List[Path]:
        """Find all C files in the given folder"""
        c_files = []
        for ext in ["*.c", "*.C"]:
            c_files.extend(folder_path.glob(ext))
        return sorted(c_files)
    
    def is_header_file(self, file_path: str) -> bool:
        """
        Check if a file is a header file based on its extension.
        Supports common header extensions: .h, .H, .hpp, .HPP, .hxx, .HXX, .h++, .H++
        """
        header_extensions = [".h", ".H", ".hpp", ".HPP", ".hxx", ".HXX", ".h++", ".H++"]
        return any(file_path.endswith(ext) for ext in header_extensions)
    
    def find_header_files(self, folder_path: Path) -> List[Path]:
        """
        Find all header files in the folder and its header subfolder.
        Looks for common header file extensions: .h, .H, .hpp, .hxx, .h++
        Note: Even if a header has a different extension, libclang will still parse it
        when it's included via #include directives in C files.
        """
        header_files = []
        
        # Common header file extensions
        header_extensions = ["*.h", "*.H", "*.hpp", "*.HPP", "*.hxx", "*.HXX", "*.h++", "*.H++"]
        
        # Headers in the folder itself
        for ext in header_extensions:
            header_files.extend(folder_path.glob(ext))
        
        # Headers in the header subfolder
        header_folder = folder_path / "header"
        if header_folder.exists():
            for ext in header_extensions:
                header_files.extend(header_folder.glob(ext))
        
        return sorted(header_files)
    
    def get_cursor_location(self, cursor: Cursor) -> str:
        """Get file location string for a cursor"""
        if cursor.location.file:
            file_path = cursor.location.file.name
            line = cursor.location.line
            col = cursor.location.column
            return f"{file_path}:{line}:{col}"
        return "unknown"
    
    def get_node_type_string(self, cursor: Cursor) -> str:
        """Get type string for a cursor"""
        try:
            if cursor.type:
                return cursor.type.spelling
            elif cursor.result_type:
                return cursor.result_type.spelling
        except:
            pass
        return ""
    
    def is_function_declaration(self, cursor: Cursor) -> bool:
        """Check if cursor is a function declaration (not definition)"""
        return (cursor.kind == CursorKind.FUNCTION_DECL and 
                cursor.is_definition() == False)
    
    def is_function_definition(self, cursor: Cursor) -> bool:
        """Check if cursor is a function definition"""
        return (cursor.kind == CursorKind.FUNCTION_DECL and 
                cursor.is_definition() == True)
    
    def parse_cursor(self, cursor: Cursor, parent_node: Optional[ASTNode] = None) -> Optional[ASTNode]:
        """
        Recursively parse a cursor and create AST nodes.
        Tracks function context to build call graph (which function calls which).
        Also handles function calls in arguments.
        """
        # Skip system headers and invalid cursors
        if cursor.location.file is None:
            return None
        
        file_path = cursor.location.file.name
        if any(skip in file_path for skip in ["/usr/include", "/System", "/Applications"]):
            return None
        
        node = None
        location = self.get_cursor_location(cursor)
        previous_function = self.current_function  # Save current function context
        
        # Handle different cursor kinds
        if cursor.kind == CursorKind.FUNCTION_DECL:
            if self.is_function_definition(cursor):
                # Function definition - set as current function context
                node = ASTNode(
                    name=cursor.spelling,
                    kind="function",
                    location=location,
                    node_type=self.get_node_type_string(cursor)
                )
                node.is_definition = True
                node.parameters = [arg.spelling for arg in cursor.get_arguments()]
                self.functions[cursor.spelling].append(node)
                
                # Set this as current function context for nested parsing
                self.current_function = node
                
            elif self.is_function_declaration(cursor):
                # Function declaration
                node = ASTNode(
                    name=cursor.spelling,
                    kind="function",
                    location=location,
                    node_type=self.get_node_type_string(cursor)
                )
                node.is_declaration = True
                node.parameters = [arg.spelling for arg in cursor.get_arguments()]
                self.functions[cursor.spelling].append(node)
        
        elif cursor.kind == CursorKind.CALL_EXPR:
            # Function call - track it in the current function context
            # Also handle nested function calls in arguments
            call_node = None
            
            # Get the called function name
            called_func_name = None
            try:
                # Try to get the function name from the cursor
                if cursor.spelling:
                    called_func_name = cursor.spelling
                else:
                    # Sometimes the name is in the referenced cursor
                    ref = cursor.referenced
                    if ref and ref.spelling:
                        called_func_name = ref.spelling
            except:
                pass
            
            if called_func_name:
                call_node = ASTNode(
                    name=called_func_name,
                    kind="function_call",
                    location=location,
                    node_type=""
                )
                
                # Track this call globally
                self.function_calls.append((call_node, called_func_name, location))
                
                # Track this call in the current function context (if we're inside a function)
                if self.current_function:
                    self.function_call_map[self.current_function].append(
                        (call_node, called_func_name, location)
                    )
                    # Add as child to current function node
                    call_node.parent = self.current_function
                    self.current_function.children.append(call_node)
            
            # Recursively parse arguments (they may contain function calls)
            for child in cursor.get_children():
                self.parse_cursor(child, call_node if call_node else parent_node)
            
            # Restore function context before returning
            self.current_function = previous_function
            return call_node
        
        elif cursor.kind == CursorKind.VAR_DECL:
            # Variable declaration
            node = ASTNode(
                name=cursor.spelling,
                kind="variable",
                location=location,
                node_type=self.get_node_type_string(cursor)
            )
            self.variables[cursor.spelling].append(node)
        
        elif cursor.kind == CursorKind.MACRO_DEFINITION:
            # Macro definition
            node = ASTNode(
                name=cursor.spelling,
                kind="macro",
                location=location,
                node_type=""
            )
            node.is_macro = True
            self.macros[cursor.spelling].append(node)
        
        elif cursor.kind == CursorKind.STRUCT_DECL:
            # Struct declaration
            if cursor.spelling:
                node = ASTNode(
                    name=cursor.spelling,
                    kind="struct",
                    location=location,
                    node_type=""
                )
                self.structs[cursor.spelling].append(node)
        
        elif cursor.kind == CursorKind.ENUM_DECL:
            # Enum declaration
            if cursor.spelling:
                node = ASTNode(
                    name=cursor.spelling,
                    kind="enum",
                    location=location,
                    node_type=""
                )
                self.enums[cursor.spelling].append(node)
        
        # Set parent-child relationship
        if node and parent_node:
            node.parent = parent_node
            parent_node.children.append(node)
        
        # Recursively parse children
        for child in cursor.get_children():
            self.parse_cursor(child, node if node else parent_node)
        
        # Restore function context if we were in a function definition
        if cursor.kind == CursorKind.FUNCTION_DECL and self.is_function_definition(cursor):
            self.current_function = previous_function
        
        return node
    
    def parse_file(self, file_path: Path) -> bool:
        """
        Parse a C file and extract AST nodes
        Returns True if successful, False otherwise
        """
        if str(file_path) in self.parsed_files:
            return True  # Already parsed
        
        print(f"Parsing: {file_path}")
        
        try:
            # Create translation unit
            args = self.get_include_args()
            tu = self.index.parse(
                str(file_path),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD | 
                        TranslationUnit.PARSE_INCOMPLETE
            )
            
            if not tu:
                print(f"  Warning: Failed to parse {file_path}")
                return False
            
            # Check for errors
            diagnostics = list(tu.diagnostics)
            if diagnostics:
                print(f"  Diagnostics for {file_path}:")
                for diag in diagnostics[:5]:  # Show first 5 errors
                    print(f"    {diag.severity}: {diag.spelling}")
            
            # Parse the root cursor
            self.parse_cursor(tu.cursor)
            
            self.parsed_files.add(str(file_path))
            return True
            
        except Exception as e:
            print(f"  Error parsing {file_path}: {e}")
            return False
    
    def find_main_function(self, folder_path: Path) -> Optional[ASTNode]:
        """Find the main function in the parsed files"""
        main_functions = self.functions.get("main", [])
        if not main_functions:
            # Also check for main with different signatures
            for func_name, nodes in self.functions.items():
                if func_name.lower() == "main":
                    main_functions = nodes
                    break
        
        if main_functions:
            # Prefer definition over declaration
            for node in main_functions:
                if node.is_definition:
                    return node
            return main_functions[0]  # Return first if no definition found
        
        return None
    
    def create_linkages(self):
        """
        Create linkages between function calls and their declarations/definitions.
        Only links if the function is declared in a header that's included in the calling file.
        This includes headers from:
        - Global header folders (like src_analysis/include) and nested subdirectories
        - Middleware folders (like moove_header) and nested subdirectories
        - Local headers in the same folder as C files
        - Headers in the header subfolder
        """
        print("\nCreating linkages...")
        
        for call_node, func_name, call_location in self.function_calls:
            # Get the file where the call is made
            call_file = call_location.split(":")[0] if ":" in call_location else ""
            
            # Find matching function declarations/definitions
            matching_functions = self.functions.get(func_name, [])
            
            if not matching_functions:
                # Check if it's a variable with the same name (should not link)
                matching_vars = self.variables.get(func_name, [])
                if matching_vars:
                    print(f"  Skipping link: '{func_name}' is a variable, not a function")
                continue
            
            # Check if any of the matching functions are in headers included by the calling file
            # For simplicity, we'll link to all matching functions
            # In a more sophisticated version, we'd check actual include relationships
            for func_node in matching_functions:
                func_file = func_node.location.split(":")[0] if ":" in func_node.location else ""
                
                # Check if function is in global header folder (including nested folders)
                is_global_header = False
                if any(keyword in func_file for keyword in ["GLOBAL_HEADER", "global_headers", "global_header", "headers"]):
                    is_global_header = True
                
                # Check if function is in middleware header folder (including nested folders)
                is_middleware_header = False
                if "moove_header" in func_file or "MIDDLE_HEADER" in func_file or "middleware" in func_file.lower():
                    is_middleware_header = True
                
                # Link if:
                # 1. Function is in a header file (.h, .hpp, .hxx, etc.) - including nested global/middleware headers OR
                # 2. Function is in the same file OR
                # 3. Function is in libapl (implementation) OR
                # 4. Function is in global header folder (regardless of extension) OR
                # 5. Function is in middleware header folder (regardless of extension)
                if (self.is_header_file(func_file) or 
                    func_file == call_file or 
                    "libapl" in func_file or
                    is_global_header or
                    is_middleware_header):
                    call_node.linked_to.append(func_node)
                    func_node.linked_to.append(call_node)
                    print(f"  Linked: {call_location} -> {func_node.location}")
    
    def build_call_graph_tree(self, function_node: ASTNode, visited: Set[str] = None, depth: int = 0, max_depth: int = 50):
        """
        Build complete call graph tree recursively.
        For each function, finds all functions it calls and adds them as children.
        Recursively builds the tree until no more nested calls are found.
        Also handles function calls in arguments.
        
        Args:
            function_node: The function node to build the tree for
            visited: Set of function names already processed (to avoid infinite loops)
            depth: Current recursion depth
            max_depth: Maximum recursion depth to prevent infinite loops
        """
        if visited is None:
            visited = set()
        
        if depth > max_depth:
            print(f"    Warning: Max depth reached for {function_node.name}")
            return
        
        # Avoid infinite loops (circular calls) - but allow same function in different call paths
        func_key = f"{function_node.name}@{function_node.location}"
        if func_key in visited:
            # Circular call detected - mark it but don't recurse further
            return
        visited.add(func_key)
        
        # Get all function calls made by this function (including in arguments)
        calls_in_function = self.function_call_map.get(function_node, [])
        
        for call_node, called_func_name, call_location in calls_in_function:
            # Find the definition of the called function
            matching_functions = self.functions.get(called_func_name, [])
            
            if not matching_functions:
                # Check if it's a variable (should not link)
                if called_func_name in self.variables:
                    continue
                # Function not found - mark as unresolved
                call_node.kind = "function_call_unresolved"
                continue
            
            # Prefer definition over declaration
            called_func_node = None
            for func_node in matching_functions:
                if func_node.is_definition:
                    called_func_node = func_node
                    break
            
            if not called_func_node:
                # Use declaration if no definition found
                called_func_node = matching_functions[0]
            
            # Check if this exact function node is already a direct child
            # (to avoid adding the same node instance twice to the same parent)
            already_direct_child = called_func_node in function_node.children
            
            if not already_direct_child:
                # Add the called function as a child
                # Note: The same function can appear in different branches of the tree
                # We prevent infinite loops using the visited set per branch
                function_node.children.append(called_func_node)
                # Store original parent if it exists (for reference, but tree allows one parent)
                if called_func_node.parent is None:
                    called_func_node.parent = function_node
                
                # Recursively build the tree for the called function
                # Use a new visited set for this branch to allow same function in different paths
                # but prevent infinite loops in circular calls
                new_visited = visited.copy()
                self.build_call_graph_tree(called_func_node, new_visited, depth + 1, max_depth)
    
    def build_complete_call_graph(self):
        """
        Build complete call graph for all functions.
        This creates a tree structure where each function has its called functions as children,
        and those called functions have their called functions as children, recursively.
        """
        print("\nBuilding complete call graph tree...")
        
        # Build call graph for all function definitions
        processed = set()
        for func_name, func_nodes in self.functions.items():
            for func_node in func_nodes:
                if func_node.is_definition:
                    func_key = f"{func_node.name}@{func_node.location}"
                    if func_key not in processed:
                        # Clear existing children that are just call nodes
                        # We'll rebuild them with actual function nodes
                        func_node.children = [
                            child for child in func_node.children 
                            if child.kind != "function_call"
                        ]
                        
                        self.build_call_graph_tree(func_node)
                        processed.add(func_key)
        
        print(f"  Built call graph for {len(processed)} function(s)")
    
    def print_ast(self, node: ASTNode, indent: int = 0, max_depth: int = 10):
        """Print AST in a tree format"""
        if indent > max_depth:
            return
        
        prefix = "  " * indent
        kind_symbol = {
            "function": "F",
            "variable": "V",
            "macro": "M",
            "struct": "S",
            "enum": "E",
            "function_call": "C",
            "function_call_unresolved": "C?"
        }
        symbol = kind_symbol.get(node.kind, "?")
        
        decl_info = ""
        if node.is_declaration:
            decl_info = " [DECL]"
        if node.is_definition:
            decl_info = " [DEF]"
        
        link_info = ""
        if node.linked_to:
            link_info = f" -> {len(node.linked_to)} link(s)"
        
        print(f"{prefix}{symbol} {node.name} ({node.kind}){decl_info}{link_info}")
        print(f"{prefix}    @ {node.location}")
        if node.node_type:
            print(f"{prefix}    Type: {node.node_type}")
        if node.parameters:
            print(f"{prefix}    Params: {', '.join(node.parameters)}")
        
        for child in node.children:
            self.print_ast(child, indent + 1, max_depth)
    
    def print_summary(self):
        """Print summary of parsed AST"""
        print("\n" + "="*80)
        print("AST SUMMARY")
        print("="*80)
        
        print(f"\nFunctions: {sum(len(nodes) for nodes in self.functions.values())}")
        for name, nodes in sorted(self.functions.items()):
            print(f"  {name}: {len(nodes)} node(s)")
            for node in nodes:
                decl_type = "DEF" if node.is_definition else "DECL"
                print(f"    [{decl_type}] {node.location}")
        
        print(f"\nVariables: {sum(len(nodes) for nodes in self.variables.values())}")
        for name, nodes in sorted(self.variables.items())[:10]:  # Show first 10
            print(f"  {name}: {len(nodes)} node(s)")
        
        print(f"\nMacros: {sum(len(nodes) for nodes in self.macros.values())}")
        for name, nodes in sorted(self.macros.items())[:10]:  # Show first 10
            print(f"  {name}: {len(nodes)} node(s)")
        
        print(f"\nStructs: {sum(len(nodes) for nodes in self.structs.values())}")
        for name, nodes in sorted(self.structs.items()):
            print(f"  {name}: {len(nodes)} node(s)")
        
        print(f"\nEnums: {sum(len(nodes) for nodes in self.enums.values())}")
        for name, nodes in sorted(self.enums.items()):
            print(f"  {name}: {len(nodes)} node(s)")
        
        print(f"\nFunction Calls: {len(self.function_calls)}")
        print(f"Parsed Files: {len(self.parsed_files)}")
    
    def generate_ast(self, folder_name: str):
        """
        Main method to generate AST for a given folder.
        Looks for folders in src_analysis/src/ structure first,
        then falls back to base_path if not found.
        """
        # First try: src_analysis/src/folder_name (expected structure)
        folder_path = self.base_path / "src_analysis" / "src" / folder_name
        
        # Fallback: try directly in base_path
        if not folder_path.exists():
            folder_path = self.base_path / folder_name
        
        if not folder_path.exists():
            print(f"Error: Folder '{folder_name}' not found in:")
            print(f"  - {self.base_path / 'src_analysis' / 'src' / folder_name}")
            print(f"  - {self.base_path / folder_name}")
            return None
        
        print(f"\n{'='*80}")
        print(f"Generating AST for folder: {folder_name}")
        print(f"{'='*80}\n")
        
        # Setup include paths
        self.setup_include_paths(folder_path)
        
        # Step 1: Parse all C files in the folder
        print("Step 1: Parsing C files...")
        c_files = self.find_c_files(folder_path)
        print(f"Found {len(c_files)} C file(s)")
        
        for c_file in c_files:
            self.parse_file(c_file)
        
        # Step 2: Parse header files in the folder
        print("\nStep 2: Parsing header files...")
        header_files = self.find_header_files(folder_path)
        print(f"Found {len(header_files)} header file(s)")
        
        for header_file in header_files:
            self.parse_file(header_file)
        
        # Step 3: Parse libapl C files if they exist
        print("\nStep 3: Parsing libapl files...")
        libapl_path = folder_path / "libapl"
        if libapl_path.exists():
            libapl_files = self.find_c_files(libapl_path)
            print(f"Found {len(libapl_files)} libapl C file(s)")
            for libapl_file in libapl_files:
                self.parse_file(libapl_file)
        else:
            print("No libapl folder found")
        
        # Step 4: Find main function
        print("\nStep 4: Finding main function...")
        main_node = self.find_main_function(folder_path)
        if main_node:
            print(f"Found main function at: {main_node.location}")
        else:
            print("Warning: Main function not found")
        
        # Step 5: Create linkages
        self.create_linkages()
        
        # Step 6: Build complete call graph tree (recursive function call hierarchy)
        self.build_complete_call_graph()
        
        # Step 7: Print AST starting from main
        print("\n" + "="*80)
        print("AST TREE (starting from main)")
        print("="*80)
        if main_node:
            self.print_ast(main_node)
        else:
            print("No main function found. Showing all functions:")
            for name, nodes in sorted(self.functions.items()):
                for node in nodes:
                    if node.is_definition:
                        self.print_ast(node)
                        break
        
        # Print summary
        self.print_summary()
        
        return main_node


def main():
    """Main entry point"""
    print("="*80)
    print("C Code AST Generator using libclang")
    print("="*80)
    
    # Get base path (current directory or provided path)
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = os.getcwd()
    
    print(f"Base path: {base_path}")
    
    # Check if libclang is available
    try:
        from clang.cindex import Index
        Index.create()
    except Exception as e:
        print(f"\nError: libclang is not properly configured.")
        print(f"Error details: {e}")
        print("\nPlease install libclang:")
        print("  macOS: brew install llvm")
        print("  Linux: sudo apt-get install libclang-dev")
        print("  Or set the library path: Config.set_library_path('/path/to/libclang')")
        return 1
    
    # Create AST generator
    generator = ASTGenerator(base_path)
    
    # Ask for folder name
    print("\nAvailable folders:")
    base = Path(base_path)
    
    # First try: look in src_analysis/src/ (expected structure)
    src_analysis_src = base / "src_analysis" / "src"
    folders = []
    if src_analysis_src.exists():
        folders = [d.name for d in src_analysis_src.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if folders:
            print(f"  Found in {src_analysis_src}:")
            for i, folder in enumerate(folders, 1):
                print(f"    {i}. {folder}")
    
    # Fallback: look in base_path
    base_folders = [d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if base_folders:
        if folders:
            print(f"\n  Also found in {base}:")
        else:
            print(f"  Found in {base}:")
        for i, folder in enumerate(base_folders, len(folders) + 1):
            print(f"    {i}. {folder}")
        folders.extend(base_folders)
    
    if not folders:
        print("  No folders found!")
    
    folder_name = input("\nEnter folder name (e.g., apl001, apl002): ").strip()
    
    if not folder_name:
        print("Error: No folder name provided")
        return 1
    
    # Generate AST
    generator.generate_ast(folder_name)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
