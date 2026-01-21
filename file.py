import os
import re
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

def parse_makefile_variables(makefile_path: str, extra_vars: Optional[Dict[str, str]] = None, max_depth: int = 8) -> Dict[str, str]:
    """
    Parse Makefile variables, following include directives (include/-include).
    
    Returns resolved variables (after $(VAR) expansion) using:
    - values defined in Makefiles
    - extra_vars overrides
    - environment variables
    """
    makefile = Path(makefile_path)
    if not makefile.exists():
        raise FileNotFoundError(f"Makefile not found: {makefile_path}")
    
    raw_vars: Dict[str, str] = {}
    resolved_vars: Dict[str, str] = {}
    extra_vars = extra_vars or {}
    visited: Set[Path] = set()
    var_pattern = re.compile(r'\$\(([^)]+)\)|\$\{([^}]+)\}')
    
    def strip_comments(line: str) -> str:
        for idx, ch in enumerate(line):
            if ch == '#' and (idx == 0 or line[idx - 1] != '\\'):
                return line[:idx]
        return line
    
    def join_continuations(lines: Iterable[str]) -> List[str]:
        joined: List[str] = []
        buffer = ""
        for line in lines:
            cleaned = strip_comments(line.rstrip("\n"))
            if not cleaned and not buffer:
                continue
            if cleaned.rstrip().endswith("\\"):
                buffer += cleaned.rstrip()[:-1] + " "
                continue
            buffer += cleaned
            if buffer.strip():
                joined.append(buffer.strip())
            buffer = ""
        if buffer.strip():
            joined.append(buffer.strip())
        return joined
    
    def resolve_var(name: str, stack: Optional[Set[str]] = None) -> str:
        if name in resolved_vars:
            return resolved_vars[name]
        if stack is None:
            stack = set()
        if name in stack:
            return ""
        stack.add(name)
        
        raw = raw_vars.get(name, extra_vars.get(name, os.environ.get(name, "")))
        if raw is None:
            raw = ""
        
        def repl(match: re.Match) -> str:
            var_name = match.group(1) or match.group(2) or ""
            return resolve_var(var_name.strip(), stack)
        
        resolved = var_pattern.sub(repl, raw)
        resolved_vars[name] = resolved
        stack.remove(name)
        return resolved
    
    def expand_string(value: str) -> str:
        if not value:
            return ""
        return var_pattern.sub(lambda m: resolve_var((m.group(1) or m.group(2) or "").strip()), value)
    
    def resolve_include_paths(include_expr: str, base_dir: Path) -> List[Path]:
        expanded = expand_string(include_expr)
        try:
            tokens = shlex.split(expanded)
        except ValueError:
            tokens = expanded.split()
        paths: List[Path] = []
        for token in tokens:
            if not token:
                continue
            token_path = Path(token)
            if not token_path.is_absolute():
                token_path = base_dir / token_path
            if any(ch in token for ch in ["*", "?", "["]):
                paths.extend(sorted(token_path.parent.glob(token_path.name)))
            else:
                paths.append(token_path)
        return paths
    
    def parse_file(path: Path, depth: int):
        if depth > max_depth or path in visited or not path.exists():
            return
        visited.add(path)
        base_dir = path.parent
        lines = join_continuations(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        for line in lines:
            if line.startswith("include ") or line.startswith("-include "):
                include_expr = line.split(None, 1)[1] if " " in line else ""
                for inc_path in resolve_include_paths(include_expr, base_dir):
                    parse_file(inc_path, depth + 1)
                continue
            match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*([:+?]?=)\s*(.*)$', line)
            if not match:
                continue
            name, op, value = match.group(1), match.group(2), match.group(3).strip()
            if op == "+=" and name in raw_vars:
                raw_vars[name] = f"{raw_vars[name]} {value}".strip()
            else:
                raw_vars[name] = value
    
    parse_file(makefile, 0)
    for name in list(raw_vars.keys()):
        resolve_var(name)
    return resolved_vars

def resolve_makefile_var_paths(makefile_path: str, var_name: str, extra_vars: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Resolve a variable to path(s), expanding includes and normalizing relative paths.
    """
    resolved = parse_makefile_variables(makefile_path, extra_vars=extra_vars)
    value = resolved.get(var_name, "")
    if not value:
        return []
    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = value.split()
    base_dir = Path(makefile_path).parent
    paths: List[str] = []
    for token in tokens:
        if not token:
            continue
        p = Path(token).expanduser()
        if not p.is_absolute():
            p = base_dir / p
        paths.append(str(p))
    return paths

def get_project_root(env_var: str = "PROJECT") -> str:
    """
    Read $(PROJECT) from environment and normalize it for string concatenation.
    """
    value = os.environ.get(env_var, "")
    if value and not value.endswith(os.sep):
        value += os.sep
    return value

def parse_makefile_variables(makefile_path: str, extra_vars: Optional[Dict[str, str]] = None, max_depth: int = 8) -> Dict[str, str]:
    """
    Parse Makefile variables, following include directives (include/-include).
    
    Returns resolved variables (after $(VAR) expansion) using:
    - values defined in Makefiles
    - extra_vars overrides
    - environment variables
    """
    makefile = Path(makefile_path)
    if not makefile.exists():
        raise FileNotFoundError(f"Makefile not found: {makefile_path}")
    
    raw_vars: Dict[str, str] = {}
    resolved_vars: Dict[str, str] = {}
    extra_vars = extra_vars or {}
    visited: Set[Path] = set()
    var_pattern = re.compile(r'\$\(([^)]+)\)|\$\{([^}]+)\}')
    
    def strip_comments(line: str) -> str:
        for idx, ch in enumerate(line):
            if ch == '#' and (idx == 0 or line[idx - 1] != '\\'):
                return line[:idx]
        return line
    
    def join_continuations(lines: Iterable[str]) -> List[str]:
        joined: List[str] = []
        buffer = ""
        for line in lines:
            cleaned = strip_comments(line.rstrip("\n"))
            if not cleaned and not buffer:
                continue
            if cleaned.rstrip().endswith("\\"):
                buffer += cleaned.rstrip()[:-1] + " "
                continue
            buffer += cleaned
            if buffer.strip():
                joined.append(buffer.strip())
            buffer = ""
        if buffer.strip():
            joined.append(buffer.strip())
        return joined
    
    def resolve_var(name: str, stack: Optional[Set[str]] = None) -> str:
        if name in resolved_vars:
            return resolved_vars[name]
        if stack is None:
            stack = set()
        if name in stack:
            return ""
        stack.add(name)
        
        raw = raw_vars.get(name, extra_vars.get(name, os.environ.get(name, "")))
        if raw is None:
            raw = ""
        
        def repl(match: re.Match) -> str:
            var_name = match.group(1) or match.group(2) or ""
            return resolve_var(var_name.strip(), stack)
        
        resolved = var_pattern.sub(repl, raw)
        resolved_vars[name] = resolved
        stack.remove(name)
        return resolved
    
    def expand_string(value: str) -> str:
        if not value:
            return ""
        return var_pattern.sub(lambda m: resolve_var((m.group(1) or m.group(2) or "").strip()), value)
    
    def resolve_include_paths(include_expr: str, base_dir: Path) -> List[Path]:
        expanded = expand_string(include_expr)
        try:
            tokens = shlex.split(expanded)
        except ValueError:
            tokens = expanded.split()
        paths: List[Path] = []
        for token in tokens:
            if not token:
                continue
            token_path = Path(token)
            if not token_path.is_absolute():
                token_path = base_dir / token_path
            if any(ch in token for ch in ["*", "?", "["]):
                paths.extend(sorted(token_path.parent.glob(token_path.name)))
            else:
                paths.append(token_path)
        return paths
    
    def parse_file(path: Path, depth: int):
        if depth > max_depth or path in visited or not path.exists():
            return
        visited.add(path)
        base_dir = path.parent
        lines = join_continuations(path.read_text(encoding="utf-8", errors="ignore").splitlines())
        for line in lines:
            if line.startswith("include ") or line.startswith("-include "):
                include_expr = line.split(None, 1)[1] if " " in line else ""
                for inc_path in resolve_include_paths(include_expr, base_dir):
                    parse_file(inc_path, depth + 1)
                continue
            match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*([:+?]?=)\s*(.*)$', line)
            if not match:
                continue
            name, op, value = match.group(1), match.group(2), match.group(3).strip()
            if op == "+=" and name in raw_vars:
                raw_vars[name] = f"{raw_vars[name]} {value}".strip()
            else:
                raw_vars[name] = value
    
    parse_file(makefile, 0)
    for name in list(raw_vars.keys()):
        resolve_var(name)
    return resolved_vars

def resolve_makefile_var_paths(makefile_path: str, var_name: str, extra_vars: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Resolve a variable to path(s), expanding includes and normalizing relative paths.
    """
    resolved = parse_makefile_variables(makefile_path, extra_vars=extra_vars)
    value = resolved.get(var_name, "")
    if not value:
        return []
    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = value.split()
    base_dir = Path(makefile_path).parent
    paths: List[str] = []
    for token in tokens:
        if not token:
            continue
        p = Path(token).expanduser()
        if not p.is_absolute():
            p = base_dir / p
        paths.append(str(p))
    return paths
