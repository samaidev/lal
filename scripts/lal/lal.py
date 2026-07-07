"""
LAL — Logic-Assembly Language (v0.2)
=====================================

A logic-native language: write a program using concept/boundary/relation
primitives, and the compiler emits specialized C code with no runtime
abstraction overhead.

v0.2 additions (this file):
  - VSA operators: bind (XOR-based), bundle (normalized sum), permute (cyclic shift)
  - MEMORY primitive: fact store with soft (similarity-weighted) lookup
  - Rule recursion: rules can call other rules
  - Multiple rules per program; compiler emits one C function per rule

Primitives:

    concept name = [v0, v1, ...]                       # a static vector
    bound   name = [d0, d1, ...]                       # a dimension mask
    memory  name = [key1: val1, key2: val2, ...]       # a fact store (keys and values are concept names)
    relate  name(a, b) = dot(a, b) @ bound_name        # masked dot product
    relate  name(a, b) = bind(a, b)                    # VSA bind: element-wise XOR (clipped to [-1,1])
    relate  name(a, b) = bundle(a, b)                  # VSA bundle: normalized sum
    relate  name(a)    = permute(a, k)                 # VSA permute: cyclic shift by k
    rule    name(args):
        var = expr                                     # use relations, ops, rule calls, recall
        output(var)

Expressions:
  - Arithmetic/logic: and, or, not, <, >, <=, >=, +, -, *
  - Relation calls:   relate_name(arg1, arg2)
  - Rule calls:       rule_name(arg)        # recursion / composition
  - Memory recall:    recall(memory_name, query_concept)   # soft lookup
  - argmax:           argmax { label: expr, label: expr, ... }

The compiler (lalc) emits specialized C code where:
  - BOUNDS are applied at compile time (unused dims deleted)
  - DOT products are fully unrolled
  - BIND/BUNDLE/PERMUTE are unrolled as element-wise ops
  - MEMORY recall is compiled to a fixed sequence of dot products + weighted sum
  - Rule calls are inlined (with recursion unrolled to a fixed depth)
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Set


# =============================================================================
# AST
# =============================================================================

@dataclass
class Concept:
    name: str
    vec: List[float]

@dataclass
class Bound:
    name: str
    dims: List[int]

@dataclass
class Memory:
    """A fact store: a list of (key_concept_name, value_concept_name) pairs.
    At runtime, recall(memory, query) returns the similarity-weighted average
    of value vectors, weighted by similarity(query, key)."""
    name: str
    entries: List[Tuple[str, str]]   # (key_name, value_name)

@dataclass
class Relate:
    name: str
    params: List[str]                # parameter names
    op: str                          # "dot", "bind", "bundle", "permute", "vadd", "vsub", "linear"
    bound_name: Optional[str]        # for dot only
    permute_k: Optional[int]         # for permute only
    # v0.7: linear layer (compiled weight matrix, with threshold pruning)
    weight_data: Optional[List[List[float]]] = None   # [out_dim][in_dim], pruned
    weight_bias: Optional[List[float]] = None         # [out_dim]
    weight_threshold: Optional[float] = None          # pruning threshold used


@dataclass
class WeightLayer:
    """v0.7: A compiled weight matrix from a trained model (e.g. GPT-2).
    The matrix is loaded at compile time, pruned by threshold, and emitted
    as a sparse specialized linear layer — NOT a generic matmul."""
    name: str
    path: str                       # path to model file
    key: str                        # weight key in the model (e.g. "h.0.mlp.c_fc")
    threshold: float                # pruning threshold: |w| < threshold → drop
    in_dim: int
    out_dim: int
    # Filled in at parse time:
    weight: Optional[List[List[float]]] = None  # [out_dim][in_dim], pruned (zeros for dropped)
    bias: Optional[List[float]] = None          # [out_dim]
    n_kept: int = 0                              # number of non-zero connections after pruning

@dataclass
class Rule:
    name: str
    params: List[str]
    body: List["Stmt"]
    outputs: List[str]
    guard: Optional["Expr"] = None  # v0.5: if present, body only runs when guard is truthy;
                                    #       otherwise rule returns the input vector unchanged.

@dataclass
class Stmt:
    var: str
    expr: "Expr"

# Expressions
@dataclass
class ECall:
    func: str                        # and|or|not|lt|gt|le|ge|add|sub|mul
    args: List["Expr"]

@dataclass
class EVar:
    name: str

@dataclass
class EFloat:
    val: float

@dataclass
class ERelateCall:
    name: str
    args: List["Expr"]

@dataclass
class ERuleCall:
    """A call to another rule: rule_name(arg). Returns the rule's first output (a vector)."""
    name: str
    args: List["Expr"]

@dataclass
class ERecall:
    """Soft lookup in a memory: recall(memory_name, query_expr).
    Returns the similarity-weighted average of value vectors."""
    memory_name: str
    query: "Expr"

@dataclass
class EArgMax:
    """argmax { label: expr, ... } — returns the index of the largest expr."""
    items: List[Tuple[str, "Expr"]]

Expr = Union[ECall, EVar, EFloat, ERelateCall, ERuleCall, ERecall, EArgMax]


# =============================================================================
# Parser
# =============================================================================

class ParseError(Exception):
    pass

def _parse_float_list(s: str) -> List[float]:
    s = s.strip().strip("[]")
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",")]

def _parse_int_list(s: str) -> List[int]:
    s = s.strip().strip("[]")
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",")]


# Embedding file cache — avoid re-reading the same file for multiple concepts.
_EMBEDDING_CACHE: Dict[str, Dict[str, List[float]]] = {}
# Directory of the .lal source file being parsed (for resolving relative embedding paths).
_CURRENT_SOURCE_DIR: Optional[str] = None
# Weight file cache: path → dict of key → (shape, flat_data)
_WEIGHT_FILE_CACHE: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {}


def _load_weight_file(path: str) -> Dict[str, Tuple[List[int], 'np.ndarray']]:
    """Load a GPW2-format weight file. Returns dict: key → (shape, numpy_array).
    Uses numpy for memory efficiency (500MB file → 500MB numpy, not 2GB Python lists)."""
    if not os.path.isabs(path):
        candidates = [path]
        if _CURRENT_SOURCE_DIR:
            candidates.append(os.path.join(_CURRENT_SOURCE_DIR, path))
        resolved = None
        for c in candidates:
            if os.path.exists(c):
                resolved = c
                break
        if resolved is None:
            resolved = os.path.abspath(path)
        path = resolved

    if path not in _WEIGHT_FILE_CACHE:
        import struct
        import numpy as np
        weights: Dict[str, Tuple[List[int], 'np.ndarray']] = {}
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GPW2":
                raise ParseError(f"not a GPW2 weight file: {path}")
            (n_tensors,) = struct.unpack("I", f.read(4))
            for _ in range(n_tensors):
                (key_len,) = struct.unpack("I", f.read(4))
                key = f.read(key_len).decode("utf-8")
                (ndim,) = struct.unpack("I", f.read(4))
                shape = [struct.unpack("I", f.read(4))[0] for _ in range(ndim)]
                n_elems = 1
                for d in shape:
                    n_elems *= d
                # Read raw bytes into numpy array (memory-efficient)
                raw = f.read(4 * n_elems)
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
                weights[key] = (shape, arr)
        _WEIGHT_FILE_CACHE[path] = weights
    return _WEIGHT_FILE_CACHE[path]


def _load_weight_matrix(path: str, key: str, threshold: float) -> Tuple[List[List[float]], List[float], int]:
    """Load a 2D weight matrix [in, out] from a GPW2 file, prune by threshold,
    and return (pruned_weight[out][in], bias[out], n_kept).
    Uses numpy for efficient loading and pruning.
    """
    import numpy as np
    weights = _load_weight_file(path)
    if key not in weights:
        raise ParseError(f"weight key {key!r} not found in {path}")
    shape, arr = weights[key]  # arr is [in, out] numpy array
    if len(shape) != 2:
        raise ParseError(f"weight {key} has shape {shape}, expected 2D")
    in_dim, out_dim = shape

    # Prune: zero out entries below threshold
    arr_pruned = arr.copy()
    arr_pruned[np.abs(arr_pruned) < threshold] = 0.0

    # Transpose to [out, in] and convert to nested list for the compiler
    w_t = arr_pruned.T.tolist()  # [out_dim][in_dim]
    n_kept = int(np.count_nonzero(arr_pruned))

    # Try to load bias: key.replace(".weight", ".bias")
    bias_key = key.replace(".weight", ".bias")
    bias = [0.0] * out_dim
    if bias_key in weights:
        bshape, barr = weights[bias_key]
        if len(bshape) == 1 and bshape[0] == out_dim:
            bias = barr.tolist()

    return w_t, bias, n_kept

def _load_word2vec(path: str, word: str, dim_hint: Optional[int] = None) -> Optional[List[float]]:
    """Load a word vector from a word2vec/GloVe text-format embedding file.

    Format: one line per word, "word v1 v2 v3 ... vN".
    The first line may be a header "vocab_size dim" (word2vec format) — we detect
    and skip it. Returns the vector as a list of floats, or None if not found.
    """
    # Resolve path: try as-is, then relative to the .lal source dir, then abspath.
    if not os.path.isabs(path):
        candidates = [path]
        if _CURRENT_SOURCE_DIR:
            candidates.append(os.path.join(_CURRENT_SOURCE_DIR, path))
        resolved = None
        for c in candidates:
            if os.path.exists(c):
                resolved = c
                break
        if resolved is None:
            # Fall back to abspath for a clear error message
            resolved = os.path.abspath(path)
        path = resolved

    if path not in _EMBEDDING_CACHE:
        emap: Dict[str, List[float]] = {}
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first = True
            for line in f:
                parts = line.rstrip("\n").split(" ")
                # Skip word2vec header line: "vocab_size dim"
                if first:
                    first = False
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        continue
                if len(parts) < 2:
                    continue
                w = parts[0]
                try:
                    v = [float(x) for x in parts[1:] if x != ""]
                except ValueError:
                    continue
                emap[w] = v
        _EMBEDDING_CACHE[path] = emap
    emap = _EMBEDDING_CACHE[path]
    if word not in emap:
        return None
    vec = emap[word]
    if dim_hint is not None and len(vec) != dim_hint:
        raise ParseError(f"word {word!r} in {path} has dim {len(vec)}, expected {dim_hint}")
    return vec

def _parse_memory_entries(s: str) -> List[Tuple[str, str]]:
    """Parse '[key1: val1, key2: val2, ...]' into a list of (key, val) tuples."""
    s = s.strip().strip("[]")
    if not s:
        return []
    entries = []
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ParseError(f"memory entry must be 'key: value', got: {pair!r}")
        k, v = pair.split(":", 1)
        entries.append((k.strip(), v.strip()))
    return entries


# ----- expression parser -----

class _ExprParser:
    """Recursive-descent parser for LAL expressions.

    Grammar:
      expr   := or_expr
      or_expr := and_expr ('or' and_expr)*
      and_expr := cmp_expr ('and' cmp_expr)*
      cmp_expr := add_expr ('<'|'>'|'<='|'>=' add_expr)?
      add_expr := mul_expr (('+'|'-') mul_expr)*
      mul_expr := unary ('*' unary)*
      unary  := '-' unary | atom
      atom   := NUMBER | NAME | NAME '(' args ')' | recall '(' NAME ',' expr ')' | argmax '{' ... '}'
      args   := expr (',' expr)*
    """
    def __init__(self, s: str):
        self.s = s
        self.i = 0
        self.n = len(s)

    def _peek(self) -> str:
        self._skip_ws()
        return self.s[self.i] if self.i < self.n else ""

    def _skip_ws(self):
        while self.i < self.n and self.s[self.i] in " \t":
            self.i += 1

    def _match(self, tok: str) -> bool:
        self._skip_ws()
        if self.s[self.i:self.i+len(tok)] == tok:
            # For alpha tokens, ensure we're not matching a prefix of a longer identifier
            if tok and tok[0].isalpha():
                end = self.i + len(tok)
                if end < self.n and (self.s[end].isalnum() or self.s[end] == "_"):
                    return False
            self.i += len(tok)
            return True
        return False

    def _expect(self, tok: str):
        if not self._match(tok):
            raise ParseError(f"expected '{tok}' at pos {self.i} in: {self.s!r}")

    def _read_ident(self) -> str:
        self._skip_ws()
        start = self.i
        while self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
            self.i += 1
        if start == self.i:
            raise ParseError(f"expected identifier at pos {self.i} in: {self.s!r}")
        return self.s[start:self.i]

    def _read_number(self) -> float:
        self._skip_ws()
        start = self.i
        while self.i < self.n and (self.s[self.i].isdigit() or self.s[self.i] in ".-+eE"):
            self.i += 1
        return float(self.s[start:self.i])

    def parse(self) -> Expr:
        e = self.parse_ternary()
        self._skip_ws()
        if self.i < self.n:
            raise ParseError(f"trailing content at pos {self.i}: {self.s[self.i:]!r}")
        return e

    def parse_ternary(self) -> Expr:
        """ternary := or_expr ('?' or_expr ':' ternary)?
        Emits ECall('if', [cond, true_expr, false_expr]).
        Right-associative. Short-circuits in both reference interpreter and C."""
        cond = self.parse_or()
        if self._match("?"):
            true_expr = self.parse_or()
            self._expect(":")
            false_expr = self.parse_ternary()  # right-assoc
            return ECall("if", [cond, true_expr, false_expr])
        return cond

    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self._match("or"):
            right = self.parse_and()
            left = ECall("or", [left, right])
        return left

    def parse_and(self) -> Expr:
        left = self.parse_cmp()
        while self._match("and"):
            right = self.parse_cmp()
            left = ECall("and", [left, right])
        return left

    def parse_cmp(self) -> Expr:
        left = self.parse_add()
        for op, name in [("<=", "le"), (">=", "ge"), ("<", "lt"), (">", "gt")]:
            if self._match(op):
                right = self.parse_add()
                return ECall(name, [left, right])
        return left

    def parse_add(self) -> Expr:
        left = self.parse_mul()
        while True:
            if self._match("+"):
                right = self.parse_mul()
                left = ECall("add", [left, right])
            elif self._match("-"):
                right = self.parse_mul()
                left = ECall("sub", [left, right])
            else:
                break
        return left

    def parse_mul(self) -> Expr:
        left = self.parse_unary()
        while True:
            if self._match("*"):
                right = self.parse_unary()
                left = ECall("mul", [left, right])
            else:
                break
        return left

    def parse_unary(self) -> Expr:
        if self._match("-"):
            inner = self.parse_unary()
            return ECall("sub", [EFloat(0.0), inner])
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        self._skip_ws()
        if self.i >= self.n:
            raise ParseError(f"unexpected end of input in: {self.s!r}")

        c = self.s[self.i]
        if c == "(":
            self._expect("(")
            e = self.parse_ternary()
            self._expect(")")
            return e

        if c.isdigit() or c == "." or (c == "-" and self.i+1 < self.n and (self.s[self.i+1].isdigit() or self.s[self.i+1] == ".")):
            return EFloat(self._read_number())

        if c.isalpha() or c == "_":
            name = self._read_ident()
            # recall(memory_name, query_expr) — special form
            if name == "recall":
                self._expect("(")
                mem_name = self._read_ident()
                self._expect(",")
                query = self.parse_ternary()
                self._expect(")")
                return ERecall(mem_name, query)
            # function call?
            if self._peek() == "(":
                self._expect("(")
                args = []
                if self._peek() != ")":
                    args.append(self.parse_ternary())
                    while self._match(","):
                        args.append(self.parse_ternary())
                self._expect(")")
                return ECall(name, args)
            return EVar(name)

        raise ParseError(f"unexpected char '{c}' at pos {self.i} in: {self.s!r}")

    def parse_argmax_block(self) -> Expr:
        """Parse 'argmax { label: expr, ... }'."""
        self._skip_ws()
        self._expect("{")
        items = []
        while True:
            label = self._read_ident()
            self._expect(":")
            expr = self.parse_ternary()
            items.append((label, expr))
            if self._match(","):
                continue
            break
        self._expect("}")
        return EArgMax(items)


def _parse_expr(s: str) -> Expr:
    return _ExprParser(s).parse()

def _parse_argmax_special(s: str) -> Expr:
    """Parse 'argmax { label: expr, ... }' from a string starting with 'argmax'."""
    s = s.strip()
    assert s.startswith("argmax"), s
    s = s[len("argmax"):].strip()
    return _ExprParser(s).parse_argmax_block()


# ----- top-level parser -----

def parse(source: str, source_path: Optional[str] = None):
    """Parse LAL source into (concepts, bounds, memories, relates, rules).

    source_path: the path to the .lal file (used to resolve relative embedding paths).
                 If None, relative paths are resolved against the current working directory.
    """
    # Stash the source directory for embedding path resolution.
    global _CURRENT_SOURCE_DIR
    _CURRENT_SOURCE_DIR = os.path.dirname(os.path.abspath(source_path)) if source_path else None

    concepts: List[Concept] = []
    bounds: List[Bound] = []
    memories: List[Memory] = []
    relates: List[Relate] = []
    rules: List[Rule] = []

    lines = source.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].split("#")[0].strip()
        if not line:
            i += 1
            continue

        # concept name = [...]   (literal vector)
        m = re.match(r"concept\s+(\w+)\s*=\s*(\[.*\])\s*$", line)
        if m:
            concepts.append(Concept(m.group(1), _parse_float_list(m.group(2))))
            i += 1
            continue

        # concept name = load_word2vec("path", "word")   (load from embedding file at compile time)
        m = re.match(r'concept\s+(\w+)\s*=\s*load_word2vec\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*(?:,\s*dim\s*=\s*(\d+))?\s*\)\s*$', line)
        if m:
            cname = m.group(1)
            path = m.group(2)
            word = m.group(3)
            dim_hint = int(m.group(4)) if m.group(4) else None
            vec = _load_word2vec(path, word, dim_hint)
            if vec is None:
                raise ParseError(f"word {word!r} not found in embedding file {path}")
            concepts.append(Concept(cname, vec))
            i += 1
            continue

        # concept name = load_glove("path", "word")   (GloVe format: same as word2vec text)
        m = re.match(r'concept\s+(\w+)\s*=\s*load_glove\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*(?:,\s*dim\s*=\s*(\d+))?\s*\)\s*$', line)
        if m:
            cname = m.group(1)
            path = m.group(2)
            word = m.group(3)
            dim_hint = int(m.group(4)) if m.group(4) else None
            vec = _load_word2vec(path, word, dim_hint)  # same format
            if vec is None:
                raise ParseError(f"word {word!r} not found in embedding file {path}")
            concepts.append(Concept(cname, vec))
            i += 1
            continue

        # bound name = [...]
        m = re.match(r"bound\s+(\w+)\s*=\s*(\[.*\])\s*$", line)
        if m:
            bounds.append(Bound(m.group(1), _parse_int_list(m.group(2))))
            i += 1
            continue

        # memory name = [k1: v1, k2: v2, ...]   (may span multiple lines until ] is found)
        m = re.match(r"memory\s+(\w+)\s*=\s*(\[[^\]]*$)", line)
        if m:
            name = m.group(1)
            rest = m.group(2)
            # Accumulate lines until the closing ] is found
            while "]" not in rest:
                i += 1
                if i >= len(lines):
                    raise ParseError(f"unterminated memory: {line!r}")
                nl = lines[i].split("#")[0].rstrip()
                if nl.strip():
                    rest += " " + nl.strip()
            # Extract the [...] part
            mb = re.match(r"\[(.*)\]\s*$", rest.strip())
            if not mb:
                raise ParseError(f"can't parse memory body: {rest!r}")
            memories.append(Memory(name, _parse_memory_entries("[" + mb.group(1) + "]")))
            i += 1
            continue

        # Single-line memory: name = [...]
        m = re.match(r"memory\s+(\w+)\s*=\s*(\[.*\])\s*$", line)
        if m:
            memories.append(Memory(m.group(1), _parse_memory_entries(m.group(2))))
            i += 1
            continue

        # relate name(params) = dot(a, b) @ bound_name
        m = re.match(r"relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*dot\(([^)]*)\)\s*(?:@\s*(\w+))?\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            dot_args = [a.strip() for a in m.group(3).split(",") if a.strip()]
            bound_name = m.group(4)
            if len(params) != 2 or len(dot_args) != 2:
                raise ParseError(f"relate dot must have 2 params and 2 args: {line!r}")
            relates.append(Relate(name, params, "dot", bound_name, None))
            i += 1
            continue

        # relate name(params) = bind(a, b)
        m = re.match(r"relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*bind\(([^)]*)\)\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            args = [a.strip() for a in m.group(3).split(",") if a.strip()]
            if len(params) != 2 or len(args) != 2:
                raise ParseError(f"relate bind must have 2 params and 2 args: {line!r}")
            relates.append(Relate(name, params, "bind", None, None))
            i += 1
            continue

        # relate name(params) = bundle(a, b)
        m = re.match(r"relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*bundle\(([^)]*)\)\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            args = [a.strip() for a in m.group(3).split(",") if a.strip()]
            if len(params) != 2 or len(args) != 2:
                raise ParseError(f"relate bundle must have 2 params and 2 args: {line!r}")
            relates.append(Relate(name, params, "bundle", None, None))
            i += 1
            continue

        # relate name(a, b) = vadd(a, b)   (vector element-wise add, v0.6)
        m = re.match(r"relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*vadd\(([^)]*)\)\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            args = [a.strip() for a in m.group(3).split(",") if a.strip()]
            if len(params) != 2 or len(args) != 2:
                raise ParseError(f"relate vadd must have 2 params and 2 args: {line!r}")
            relates.append(Relate(name, params, "vadd", None, None))
            i += 1
            continue

        # relate name(a, b) = vsub(a, b)   (vector element-wise subtract, v0.6)
        m = re.match(r"relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*vsub\(([^)]*)\)\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            args = [a.strip() for a in m.group(3).split(",") if a.strip()]
            if len(params) != 2 or len(args) != 2:
                raise ParseError(f"relate vsub must have 2 params and 2 args: {line!r}")
            relates.append(Relate(name, params, "vsub", None, None))
            i += 1
            continue

        # relate name(x) = linear(x, "weights.bin", "h.0.mlp.c_fc", threshold=0.1)
        # v0.7: compiled linear layer — loads weight at compile time, prunes by threshold
        m = re.match(r'relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*linear\(\s*(\w+)\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*threshold\s*=\s*([0-9.eE+-]+)\s*\)\s*$', line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            x_arg = m.group(3)
            path = m.group(4)
            wkey = m.group(5)
            threshold = float(m.group(6))
            if len(params) != 1 or params[0] != x_arg:
                raise ParseError(f"relate linear param must match arg: {line!r}")
            # Load and prune the weight now (compile time).
            # Auto-append ".weight" if the key doesn't have it.
            if not wkey.endswith(".weight") and not wkey.endswith(".bias"):
                wkey_full = wkey + ".weight"
            else:
                wkey_full = wkey
            w_t, bias, n_kept = _load_weight_matrix(path, wkey_full, threshold)
            out_dim = len(w_t)
            in_dim = len(w_t[0]) if out_dim > 0 else 0
            relates.append(Relate(name, params, "linear", None, None,
                                  weight_data=w_t, weight_bias=bias, weight_threshold=threshold))
            print(f"[*] linear {name}: {wkey} [{in_dim}→{out_dim}], threshold={threshold}, "
                  f"kept {n_kept}/{in_dim*out_dim} ({100*n_kept/(in_dim*out_dim):.1f}%)", flush=True)
            i += 1
            continue

        # relate name(a) = permute(a, k)
        m = re.match(r"relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*permute\(([^)]*)\)\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            args = [a.strip() for a in m.group(3).split(",") if a.strip()]
            if len(params) != 1 or len(args) != 2:
                raise ParseError(f"relate permute must have 1 param and 2 args (a, k): {line!r}")
            try:
                k = int(args[1])
            except ValueError:
                raise ParseError(f"permute k must be an integer: {line!r}")
            relates.append(Relate(name, params, "permute", None, k))
            i += 1
            continue

        # rule name(params):                    (no guard)
        # rule name(params) | guard_expr:       (with guard — v0.5)
        m = re.match(r"rule\s+(\w+)\s*\(([^)]*)\)\s*(?:\|\s*(.+?)\s*)?:\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            guard_str = m.group(3)
            guard_expr = _parse_expr(guard_str) if guard_str else None
            body: List[Stmt] = []
            outputs: List[str] = []
            i += 1
            while i < len(lines):
                bline = lines[i].split("#")[0]
                if not bline.strip():
                    i += 1
                    continue
                if not (bline.startswith("    ") or bline.startswith("\t")):
                    break
                bline = bline.strip()

                # output(...)
                mout = re.match(r"output\(([^)]*)\)\s*$", bline)
                if mout:
                    outputs = [o.strip() for o in mout.group(1).split(",") if o.strip()]
                    i += 1
                    continue

                # var = expr
                mstmt = re.match(r"(\w+)\s*=\s*(.*)$", bline)
                if mstmt:
                    var = mstmt.group(1)
                    rest = mstmt.group(2).strip()
                    # argmax special form may span multiple lines
                    if rest.startswith("argmax"):
                        while "}" not in rest:
                            i += 1
                            if i >= len(lines):
                                raise ParseError(f"unterminated argmax: {bline!r}")
                            nl = lines[i].split("#")[0].rstrip()
                            if nl.strip():
                                rest += " " + nl.strip()
                        expr = _parse_argmax_special(rest)
                    else:
                        expr = _parse_expr(rest)
                    body.append(Stmt(var, expr))
                    i += 1
                    continue

                raise ParseError(f"can't parse rule body line: {bline!r}")

            rules.append(Rule(name, params, body, outputs, guard_expr))
            continue

        raise ParseError(f"can't parse line: {line!r}")

    # Post-process: convert ECall(func=relate_name, ...) to ERelateCall,
    # and ECall(func=rule_name, ...) to ERuleCall. Also process rule guards.
    relate_names = {r.name for r in relates}
    rule_names = {r.name for r in rules}
    for rule in rules:
        for stmt in rule.body:
            stmt.expr = _rewrite_calls(stmt.expr, relate_names, rule_names)
        if rule.guard is not None:
            rule.guard = _rewrite_calls(rule.guard, relate_names, rule_names)

    return concepts, bounds, memories, relates, rules


def _rewrite_calls(expr, relate_names, rule_names):
    if isinstance(expr, ECall):
        if expr.func in relate_names:
            return ERelateCall(expr.func, [_rewrite_calls(a, relate_names, rule_names) for a in expr.args])
        if expr.func in rule_names:
            return ERuleCall(expr.func, [_rewrite_calls(a, relate_names, rule_names) for a in expr.args])
        return ECall(expr.func, [_rewrite_calls(a, relate_names, rule_names) for a in expr.args])
    if isinstance(expr, ERecall):
        return ERecall(expr.memory_name, _rewrite_calls(expr.query, relate_names, rule_names))
    if isinstance(expr, EArgMax):
        return EArgMax([(l, _rewrite_calls(e, relate_names, rule_names)) for l, e in expr.items])
    return expr


# =============================================================================
# Reference interpreter (Python) — for verification
# =============================================================================

def run_reference(concepts, bounds, memories, relates, rules, rule_name, input_vec):
    """Run a rule with the given input vector. Returns the env dict."""
    concept_map = {c.name: c.vec for c in concepts}
    bound_map = {b.name: b.dims for b in bounds}
    memory_map = {m.name: m for m in memories}
    relate_map = {r.name: r for r in relates}
    # v0.6: rule_map now maps name → list of rule variants (for pattern matching).
    rule_groups: Dict[str, List[Rule]] = {}
    for r in rules:
        rule_groups.setdefault(r.name, []).append(r)
    rule_variants = rule_groups.get(rule_name, [])
    if not rule_variants:
        raise RuntimeError(f"rule {rule_name} not found")

    env: Dict[str, object] = {}

    def _vsa_bind(a, b):
        """VSA bind: element-wise XOR-like op on [-1,1] vectors.
        Using a*b sign-based binding: bind(a,b)[i] = sign(a[i]*b[i]) * min(|a[i]|,|b[i]|)
        For values in [0,1] this is just a[i]*b[i]; for [-1,1] we use sign-magnitude.
        We use the simplest form: element-wise product (Plate's HRR-style binding is
        circular convolution; we use element-wise for simplicity)."""
        return [a[i] * b[i] for i in range(len(a))]

    def _vsa_bundle(a, b):
        """VSA bundle: normalized sum. Output in [-1, 1]."""
        s = [a[i] + b[i] for i in range(len(a))]
        max_abs = max(abs(x) for x in s) if s else 1.0
        if max_abs < 1e-9:
            return s
        return [x / max_abs for x in s]

    def _vsa_permute(a, k):
        """VSA permute: cyclic shift by k."""
        n = len(a)
        return [a[(i - k) % n] for i in range(n)]

    def ev(e):
        if isinstance(e, EFloat):
            return e.val
        if isinstance(e, EVar):
            if e.name in env:
                return env[e.name]
            if e.name in concept_map:
                return concept_map[e.name]
            raise RuntimeError(f"unknown var: {e.name}")
        if isinstance(e, ECall):
            if e.func == "dot":
                raise RuntimeError("bare dot — must be inside a relate")
            # 'if' (ternary) must short-circuit: only evaluate the taken branch.
            # This is critical for recursion — the false branch may contain a
            # recursive call that would loop forever if evaluated eagerly.
            if e.func == "if":
                cond = ev(e.args[0])
                if float(cond) != 0.0:
                    return ev(e.args[1])
                else:
                    return ev(e.args[2])
            args = [ev(a) for a in e.args]
            if e.func == "and":
                return float(args[0]) * float(args[1])
            if e.func == "or":
                a, b = float(args[0]), float(args[1])
                return a + b - a*b
            if e.func == "not":
                return 1.0 - float(args[0])
            if e.func == "lt":
                return 1.0 if args[0] < args[1] else 0.0
            if e.func == "gt":
                return 1.0 if args[0] > args[1] else 0.0
            if e.func == "le":
                return 1.0 if args[0] <= args[1] else 0.0
            if e.func == "ge":
                return 1.0 if args[0] >= args[1] else 0.0
            if e.func == "add":
                return float(args[0]) + float(args[1])
            if e.func == "sub":
                return float(args[0]) - float(args[1])
            if e.func == "mul":
                return float(args[0]) * float(args[1])
            raise RuntimeError(f"unknown func: {e.func}")
        if isinstance(e, ERelateCall):
            r = relate_map[e.name]
            arg_vals = [ev(a) for a in e.args]
            if r.op == "dot":
                a, b = arg_vals[0], arg_vals[1]
                if r.bound_name:
                    dims = bound_map[r.bound_name]
                    return sum(a[d] * b[d] for d in dims)
                return sum(x*y for x, y in zip(a, b))
            if r.op == "bind":
                a, b = arg_vals[0], arg_vals[1]
                return _vsa_bind(a, b)
            if r.op == "bundle":
                a, b = arg_vals[0], arg_vals[1]
                return _vsa_bundle(a, b)
            if r.op == "permute":
                a = arg_vals[0]
                return _vsa_permute(a, r.permute_k)
            if r.op == "vadd":
                a, b = arg_vals[0], arg_vals[1]
                return [a[i] + b[i] for i in range(len(a))]
            if r.op == "vsub":
                a, b = arg_vals[0], arg_vals[1]
                return [a[i] - b[i] for i in range(len(a))]
            if r.op == "linear":
                # Compiled linear layer: y[j] = sum(x[i] * w[j][i]) + bias[j]
                # w is already pruned (zeros for dropped connections)
                x = arg_vals[0]
                w = r.weight_data
                bias = r.weight_bias
                out_dim = len(w)
                in_dim = len(w[0]) if out_dim > 0 else 0
                y = []
                for j in range(out_dim):
                    s = bias[j] if bias else 0.0
                    wj = w[j]
                    for i in range(in_dim):
                        wi = wj[i]
                        if wi != 0.0:  # skip pruned
                            s += x[i] * wi
                    y.append(s)
                return y
            raise RuntimeError(f"unknown relate op: {r.op}")
        if isinstance(e, ERuleCall):
            # Recursively run the called rule with the arg as input
            arg_val = ev(e.args[0])
            sub_env = run_reference(concepts, bounds, memories, relates, rules, e.name, arg_val)
            # Return the first output of the sub-rule (use first variant's output name)
            sub_variants = rule_groups.get(e.name, [])
            if sub_variants and sub_variants[0].outputs:
                return sub_env[sub_variants[0].outputs[0]]
            raise RuntimeError(f"rule {e.name} has no outputs")
        if isinstance(e, ERecall):
            mem = memory_map[e.memory_name]
            query = ev(e.query)
            # Soft lookup: weighted average of value vectors, weighted by dot(query, key)
            weighted_sum = [0.0] * len(concept_map[mem.entries[0][1]])
            total_weight = 0.0
            for key_name, val_name in mem.entries:
                key_vec = concept_map[key_name]
                val_vec = concept_map[val_name]
                w = sum(query[d] * key_vec[d] for d in range(len(query)))
                # Use softmax-like weighting: exponentiate positive similarities
                w = max(0.0, w)  # only positive similarities contribute
                for d in range(len(weighted_sum)):
                    weighted_sum[d] += w * val_vec[d]
                total_weight += w
            if total_weight < 1e-9:
                return weighted_sum  # all zeros
            return [x / total_weight for x in weighted_sum]
        if isinstance(e, EArgMax):
            scores = [(label, ev(expr)) for label, expr in e.items]
            best = max(scores, key=lambda x: x[1])
            return best[0]
        raise RuntimeError(f"can't eval: {e}")

    # v0.6: Try each variant in order. The first whose guard is truthy (or which
    # has no guard) runs its body. If no variant matches, return input unchanged.
    chosen_env = None
    for variant in rule_variants:
        # Set up env for this variant
        env_v = {variant.params[0]: input_vec}
        # Override the env that ev() uses — we need ev to see the variant's params.
        # Simplest: clear and reset env.
        env.clear()
        env.update(env_v)
        # Check guard
        if variant.guard is not None:
            guard_val = ev(variant.guard)
            if float(guard_val) == 0.0:
                continue  # guard failed, try next variant
        # Guard passed (or no guard) — run body
        for stmt in variant.body:
            env[stmt.var] = ev(stmt.expr)
        chosen_env = dict(env)
        break
    if chosen_env is None:
        # No variant matched: passthrough input as first output
        env.clear()
        if rule_variants[0].outputs:
            env[rule_variants[0].outputs[0]] = input_vec
        chosen_env = dict(env)
    # Restore chosen env into env (for caller)
    env.clear()
    env.update(chosen_env)
    return env


# =============================================================================
# Compiler — AST → specialized C
# =============================================================================

# Global set of SIMD dot-product widths needed in the current compilation.
# Reset at the start of each compile_to_c() call.
_SIMD_HELPERS_NEEDED: Set[int] = set()
# Global quantization mode: None (float32) or "int8".
_QUANTIZE_MODE = None
def compile_to_c(concepts, bounds, memories, relates, rules, rule_name, max_recursion_depth=2, quantize=None) -> str:
    """Compile a rule (and any rules it calls) to specialized C.

    quantize: None for float32, or "int8" to store concept vectors as int8 + scale
              (4x smaller .rodata, with runtime dequantization in the dot helper).
    """
    global _SIMD_HELPERS_NEEDED, _QUANTIZE_MODE
    _SIMD_HELPERS_NEEDED = set()  # reset for this compilation
    _QUANTIZE_MODE = quantize

    concept_map = {c.name: c.vec for c in concepts}
    bound_map = {b.name: b.dims for b in bounds}
    memory_map = {m.name: m for m in memories}
    relate_map = {r.name: r for r in relates}
    # v0.6: rule_map maps name → list of variants (for pattern matching).
    # For backward compat with code that does rule_map[name], we keep a
    # rule_map pointing to the FIRST variant of each name.
    rule_groups: Dict[str, List[Rule]] = {}
    for r in rules:
        rule_groups.setdefault(r.name, []).append(r)
    rule_map = {name: variants[0] for name, variants in rule_groups.items()}
    concept_names = set(concept_map.keys())

    # Find which rules we need to emit (all transitively-reachable rule names).
    rules_to_emit = _collect_called_rules(rule_map[rule_name], rule_map, relate_map, max_recursion_depth)
    rules_to_emit.add(rule_name)

    # Determine the vector dimension (assume all concepts have the same dim).
    vec_dim = len(concepts[0].vec) if concepts else 0

    lines = []
    lines.append("/*")
    lines.append(" * Generated by lalc v0.3 — the LAL compiler.")
    lines.append(" *")
    lines.append(f" * Entry rule: {rule_name}")
    lines.append(" * Vector dim: " + str(vec_dim))
    lines.append(" * Rules emitted (all transitively-reachable; rule calls are real C calls):")
    for rn in sorted(rules_to_emit):
        lines.append(f" *   - {rn}")
    lines.append(" *")
    lines.append(" * Specializations applied at compile time:")
    lines.append(" *   - BOUND: only used dimensions retained in concept vectors")
    lines.append(" *   - DOT: fully unrolled (scalar) or SIMD (AVX2/NEON) when width >= 8")
    lines.append(" *   - BIND/BUNDLE/PERMUTE: element-wise, fully unrolled")
    lines.append(" *   - MEMORY recall: compiled to fixed dot products + weighted sum")
    lines.append(" *   - Rule calls: real C function calls (supports true recursion)")
    lines.append(" *   - argmax: flat comparison chain, no dispatch")
    lines.append(" */")
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("#include <string.h>")
    lines.append("#include <stdint.h>")  # int8_t for quantization
    lines.append("#if defined(__AVX2__) || defined(__AVX__)")
    lines.append("#include <immintrin.h>")
    lines.append("#elif defined(__ARM_NEON) || defined(__ARM_NEON__)")
    lines.append("#include <arm_neon.h>")
    lines.append("#endif")
    lines.append("")

    # Emit full-dim concept arrays — but ONLY for concepts that are actually used
    # by VSA ops (bind/bundle/permute), recall, unbounded dots, or as memory keys/values.
    # Concepts only used in bounded dots don't need full arrays (they use masked arrays).
    concepts_needing_full = set()
    for rn in rules_to_emit:
        r = rule_map[rn]
        for stmt in r.body:
            _collect_concepts_needing_full(stmt.expr, relate_map, concept_names, concepts_needing_full, memory_map)
    # Also: memory keys/values always need full arrays (recall uses them).
    for mem in memories:
        for key_name, val_name in mem.entries:
            concepts_needing_full.add(key_name)
            concepts_needing_full.add(val_name)

    if concepts_needing_full:
        lines.append("/* === Concept vectors (full dim, used by VSA ops and MEMORY recall) ===")
        lines.append(" * Only concepts referenced by bind/bundle/permute/recall/unbounded-dot are emitted. */")
        for c in sorted(concepts, key=lambda x: x.name):
            if c.name not in concepts_needing_full:
                continue
            if _QUANTIZE_MODE == "int8":
                q, scale = _quantize_int8(c.vec)
                arr = ", ".join(str(x) for x in q)
                lines.append(f"static const int8_t concept_{c.name}_q[{len(c.vec)}] = {{{arr}}};")
                lines.append(f"static const float concept_{c.name}_scale = {scale:.8f}f;")
                # Also emit a float version for VSA ops / recall (which need float vectors).
                arr_f = ", ".join(f"{v:.6f}f" for v in c.vec)
                lines.append(f"static const float concept_{c.name}[{len(c.vec)}] = {{{arr_f}}};")
            elif _QUANTIZE_MODE == "int4":
                q, scale = _quantize_int4(c.vec)
                packed = _pack_nibbles(q)
                n_packed = len(packed)
                arr = ", ".join(str(x) for x in packed)
                lines.append(f"static const uint8_t concept_{c.name}_q4[{n_packed}] = {{{arr}}};")
                lines.append(f"static const float concept_{c.name}_scale = {scale:.8f}f;")
                # Also emit a float version for VSA ops / recall (which need float vectors).
                arr_f = ", ".join(f"{v:.6f}f" for v in c.vec)
                lines.append(f"static const float concept_{c.name}[{len(c.vec)}] = {{{arr_f}}};")
            else:
                arr = ", ".join(f"{v:.6f}f" for v in c.vec)
                lines.append(f"static const float concept_{c.name}[{len(c.vec)}] = {{{arr}}};")
        lines.append("")

    # Emit masked concept arrays for bounded dot products.
    # Collect (concept, bound) pairs used in any rule being emitted.
    used_concept_bound_pairs = set()
    for rn in rules_to_emit:
        r = rule_map[rn]
        for stmt in r.body:
            _collect_used_pairs(stmt.expr, relate_map, concept_names, used_concept_bound_pairs, rule_map, max_recursion_depth)
    if used_concept_bound_pairs:
        lines.append("/* === Masked concept vectors (BOUND applied at compile time) === */")
        for (cname, bname) in sorted(used_concept_bound_pairs):
            cvec = concept_map[cname]
            dims = bound_map[bname]
            vals = [cvec[d] for d in dims]
            if _QUANTIZE_MODE == "int8":
                q, scale = _quantize_int8(vals)
                arr = ", ".join(str(x) for x in q)
                lines.append(f"static const int8_t {cname}__{bname}_q[{len(dims)}] = {{{arr}}};")
                lines.append(f"static const float {cname}__{bname}_scale = {scale:.8f}f;")
            elif _QUANTIZE_MODE == "int4":
                q, scale = _quantize_int4(vals)
                packed = _pack_nibbles(q)
                n_packed = len(packed)
                arr = ", ".join(str(x) for x in packed)
                lines.append(f"static const uint8_t {cname}__{bname}_q4[{n_packed}] = {{{arr}}};")
                lines.append(f"static const float {cname}__{bname}_scale = {scale:.8f}f;")
            else:
                arr = ", ".join(f"{v:.6f}f" for v in vals)
                lines.append(f"static const float {cname}__{bname}[{len(dims)}] = {{{arr}}};")
        lines.append("")

    # Pre-scan all rule bodies AND guards to collect SIMD dot widths needed.
    for rn in rules_to_emit:
        r = rule_map[rn]
        for stmt in r.body:
            _collect_simd_widths(stmt.expr, relate_map, concept_names, bound_map, concept_map, _SIMD_HELPERS_NEEDED)
        if r.guard is not None:
            _collect_simd_widths(r.guard, relate_map, concept_names, bound_map, concept_map, _SIMD_HELPERS_NEEDED)

    # Emit SIMD dot-product helpers (only for widths that were used).
    if _SIMD_HELPERS_NEEDED:
        lines.append("/* === SIMD dot-product helpers (AVX2 on x86_64, NEON on arm64) ===")
        lines.append(" * Compiled in only when the target supports them; scalar fallback otherwise. */")
        for width in sorted(_SIMD_HELPERS_NEEDED):
            lines.extend(_emit_simd_dot_helper(width))
            if _QUANTIZE_MODE == "int8":
                lines.extend(_emit_simd_qdot_helper(width))
            elif _QUANTIZE_MODE == "int4":
                lines.extend(_emit_simd_q4dot_helper(width))
        lines.append("")

    # Forward-declare all rules being emitted.
    # v0.6: For multi-rule pattern matching, we emit:
    #   - rule_<name>_v<index>(params, outputs) for each variant
    #   - rule_<name>(params, outputs) as a dispatcher that tries variants in order
    lines.append("/* === Forward declarations === */")
    for rn in sorted(rules_to_emit):
        variants = rule_groups.get(rn, [rule_map[rn]])
        for idx, r in enumerate(variants):
            params_sig = ", ".join(f"const float* {p}" for p in r.params)
            out_params = []
            for out_name in r.outputs:
                matching = [s for s in r.body if s.var == out_name]
                if matching and isinstance(matching[0].expr, EArgMax):
                    out_params.append(f"int* out_{out_name}_idx")
                else:
                    out_params.append(f"float* out_{out_name}")
            out_sig = ", ".join(out_params)
            if out_sig:
                lines.append(f"void rule_{rn}_v{idx}({params_sig}, {out_sig});")
            else:
                lines.append(f"void rule_{rn}_v{idx}({params_sig});")
        # Dispatcher declaration
        r0 = variants[0]
        params_sig = ", ".join(f"const float* {p}" for p in r0.params)
        out_params = []
        for out_name in r0.outputs:
            matching = [s for s in r0.body if s.var == out_name]
            if matching and isinstance(matching[0].expr, EArgMax):
                out_params.append(f"int* out_{out_name}_idx")
            else:
                out_params.append(f"float* out_{out_name}")
        out_sig = ", ".join(out_params)
        if out_sig:
            lines.append(f"void rule_{rn}({params_sig}, {out_sig});")
        else:
            lines.append(f"void rule_{rn}({params_sig});")
    lines.append("")

    # Emit each rule variant as a C function, plus a dispatcher per rule name.
    for rn in sorted(rules_to_emit):
        variants = rule_groups.get(rn, [rule_map[rn]])
        for idx, r in enumerate(variants):
            # Emit the variant, but with a mangled name.
            lines.extend(_emit_rule(r, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rules_to_emit, max_recursion_depth, name_suffix=f"_v{idx}"))
            lines.append("")
        # Emit the dispatcher for this rule name.
        lines.extend(_emit_rule_dispatcher(rn, variants, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rules_to_emit, max_recursion_depth, vec_dim))
        lines.append("")

    # main() — calls the entry rule.
    lines.append("int main(int argc, char** argv) {")
    lines.append(f"    float q[{vec_dim}];")
    lines.append(f"    if (argc < {vec_dim+1}) {{")
    lines.append(f'        fprintf(stderr, "usage: {rn} v0 v1 ... v{vec_dim-1}\\n");')
    lines.append("        return 1;")
    lines.append("    }")
    lines.append(f"    for (int i = 0; i < {vec_dim}; i++) q[i] = (float)atof(argv[i+1]);")
    entry_rule = rule_map[rule_name]
    # Declare output vars: int for argmax, float[vec_dim] for vectors.
    for n in entry_rule.outputs:
        matching = [s for s in entry_rule.body if s.var == n]
        if matching and isinstance(matching[0].expr, EArgMax):
            lines.append(f"    int out_{n}_idx = -1;")
        else:
            lines.append(f"    float out_{n}[{vec_dim}] = {{0}};")
    # Build call args matching the rule's signature.
    out_args_list = []
    for n in entry_rule.outputs:
        matching = [s for s in entry_rule.body if s.var == n]
        if matching and isinstance(matching[0].expr, EArgMax):
            out_args_list.append(f"&out_{n}_idx")
        else:
            out_args_list.append(f"out_{n}")
    out_args = (", " + ", ".join(out_args_list)) if out_args_list else ""
    lines.append(f"    rule_{rule_name}(q{out_args});")
    # Print outputs.
    for n in entry_rule.outputs:
        matching = [s for s in entry_rule.body if s.var == n]
        if matching and isinstance(matching[0].expr, EArgMax):
            lines.append(f'    printf("%d\\n", out_{n}_idx);')
        else:
            lines.append(f'    printf("%f\\n", out_{n}[0]);')
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _collect_called_rules(rule, rule_map, relate_map, max_depth=None, seen=None, depth=0):
    """Find all rules transitively called from this rule.

    In v0.3, rule calls are real C function calls (not inlined), so we emit ALL
    transitively-reachable rules with no depth limit. The max_depth parameter is
    kept for API compatibility but ignored.
    """
    if seen is None:
        seen = set()
    for stmt in rule.body:
        for er in _find_rule_calls(stmt.expr, relate_map):
            if er not in seen:
                seen.add(er)
                _collect_called_rules(rule_map[er], rule_map, relate_map, max_depth, seen, depth+1)
    return seen


def _find_rule_calls(expr, relate_map):
    """Yield rule names called from this expression."""
    if isinstance(expr, ERuleCall):
        yield expr.name
        for a in expr.args:
            yield from _find_rule_calls(a, relate_map)
    if isinstance(expr, ECall):
        for a in expr.args:
            yield from _find_rule_calls(a, relate_map)
    if isinstance(expr, ERelateCall):
        for a in expr.args:
            yield from _find_rule_calls(a, relate_map)
    if isinstance(expr, ERecall):
        yield from _find_rule_calls(expr.query, relate_map)
    if isinstance(expr, EArgMax):
        for _, e in expr.items:
            yield from _find_rule_calls(e, relate_map)


def _collect_used_pairs(expr, relate_map, concept_names, used, rule_map, max_depth):
    """Collect (concept, bound) pairs used in bounded dot products within this expr."""
    if isinstance(expr, ERelateCall):
        r = relate_map[expr.name]
        if r.op == "dot" and r.bound_name:
            for a in expr.args:
                if isinstance(a, EVar) and a.name in concept_names:
                    used.add((a.name, r.bound_name))
        for a in expr.args:
            _collect_used_pairs(a, relate_map, concept_names, used, rule_map, max_depth)
    if isinstance(expr, ECall):
        for a in expr.args:
            _collect_used_pairs(a, relate_map, concept_names, used, rule_map, max_depth)
    if isinstance(expr, EArgMax):
        for _, e in expr.items:
            _collect_used_pairs(e, relate_map, concept_names, used, rule_map, max_depth)
    if isinstance(expr, ERecall):
        _collect_used_pairs(expr.query, relate_map, concept_names, used, rule_map, max_depth)


def _collect_param_bound_pairs(expr, relate_map, concept_names, rule_params, used):
    """Collect (param_name, bound_name) pairs used in bounded dot products where the
    non-concept arg is a rule parameter. These need masked arrays emitted at function entry."""
    if isinstance(expr, ERelateCall):
        r = relate_map[expr.name]
        if r.op == "dot" and r.bound_name:
            for a in expr.args:
                if isinstance(a, EVar) and a.name in rule_params:
                    used.add((a.name, r.bound_name))
        for a in expr.args:
            _collect_param_bound_pairs(a, relate_map, concept_names, rule_params, used)
    if isinstance(expr, ECall):
        for a in expr.args:
            _collect_param_bound_pairs(a, relate_map, concept_names, rule_params, used)
    if isinstance(expr, EArgMax):
        for _, e in expr.items:
            _collect_param_bound_pairs(e, relate_map, concept_names, rule_params, used)
    if isinstance(expr, ERecall):
        _collect_param_bound_pairs(expr.query, relate_map, concept_names, rule_params, used)


def _collect_concepts_needing_full(expr, relate_map, concept_names, needed, memory_map):
    """Collect concept names that need full-dim arrays (used by VSA ops, vadd/vsub, recall, or unbounded dot)."""
    if isinstance(expr, ERelateCall):
        r = relate_map[expr.name]
        if r.op in ("bind", "bundle", "permute", "vadd", "vsub", "linear"):
            for a in expr.args:
                if isinstance(a, EVar) and a.name in concept_names:
                    needed.add(a.name)
        if r.op == "dot" and not r.bound_name:
            # Unbounded dot needs full arrays
            for a in expr.args:
                if isinstance(a, EVar) and a.name in concept_names:
                    needed.add(a.name)
        for a in expr.args:
            _collect_concepts_needing_full(a, relate_map, concept_names, needed, memory_map)
    if isinstance(expr, ECall):
        for a in expr.args:
            _collect_concepts_needing_full(a, relate_map, concept_names, needed, memory_map)
    if isinstance(expr, EArgMax):
        for _, e in expr.items:
            _collect_concepts_needing_full(e, relate_map, concept_names, needed, memory_map)
    if isinstance(expr, ERecall):
        _collect_concepts_needing_full(expr.query, relate_map, concept_names, needed, memory_map)
        if expr.memory_name in memory_map:
            for key_name, val_name in memory_map[expr.memory_name].entries:
                needed.add(key_name)
                needed.add(val_name)


def _collect_simd_widths(expr, relate_map, concept_names, bound_map, concept_map, needed):
    """Pre-scan an expression to find dot-product widths that will need SIMD helpers."""
    if isinstance(expr, ERelateCall):
        r = relate_map[expr.name]
        if r.op == "dot":
            # Find the concept to determine width
            for a in expr.args:
                if isinstance(a, EVar) and a.name in concept_names:
                    if r.bound_name:
                        width = len(bound_map[r.bound_name])
                    else:
                        width = len(concept_map[a.name])
                    if width >= 8:
                        needed.add(width)
                    break
        for a in expr.args:
            _collect_simd_widths(a, relate_map, concept_names, bound_map, concept_map, needed)
    if isinstance(expr, ECall):
        for a in expr.args:
            _collect_simd_widths(a, relate_map, concept_names, bound_map, concept_map, needed)
    if isinstance(expr, EArgMax):
        for _, e in expr.items:
            _collect_simd_widths(e, relate_map, concept_names, bound_map, concept_map, needed)
    if isinstance(expr, ERecall):
        _collect_simd_widths(expr.query, relate_map, concept_names, bound_map, concept_map, needed)


def _emit_simd_dot_helper(width: int) -> List[str]:
    """Emit a SIMD dot-product helper for vectors of the given width.

    Uses AVX2 on x86_64 (8 floats per __m256), NEON on arm64 (4 floats per float32x4_t).
    Falls back to scalar unrolled on unsupported targets.
    Width must be >= 8 and a multiple of 4 (we only call this for width >= 8).
    """
    lines = []
    lines.append(f"/* dot product of two float[{width}] — SIMD-accelerated */")
    lines.append(f"static inline float _lal_dot_simd_{width}(const float* a, const float* b) {{")
    lines.append(f"    float result = 0.0f;")

    n_avx = width // 8
    rem_after_avx = width % 8
    n_neon = rem_after_avx // 4
    n_scalar = rem_after_avx % 4

    # AVX path (preferred on x86_64)
    lines.append(f"#if defined(__AVX2__) || defined(__AVX__)")
    if n_avx > 0:
        lines.append(f"    {{")
        lines.append(f"        __m256 acc = _mm256_setzero_ps();")
        for k in range(n_avx):
            lines.append(f"        __m256 va_{k} = _mm256_loadu_ps(a + {k*8});")
            lines.append(f"        __m256 vb_{k} = _mm256_loadu_ps(b + {k*8});")
            lines.append(f"        acc = _mm256_fmadd_ps(va_{k}, vb_{k}, acc);")
        lines.append(f"        __m128 lo = _mm256_castps256_ps128(acc);")
        lines.append(f"        __m128 hi = _mm256_extractf128_ps(acc, 1);")
        lines.append(f"        __m128 s = _mm_add_ps(lo, hi);")
        lines.append(f"        s = _mm_hadd_ps(s, s);")
        lines.append(f"        s = _mm_hadd_ps(s, s);")
        lines.append(f"        result = _mm_cvtss_f32(s);")
        # Scalar tail (remainder after 8-float blocks)
        offset = n_avx * 8
        for k in range(rem_after_avx):
            lines.append(f"        result += a[{offset+k}] * b[{offset+k}];")
        lines.append(f"    }}")
    else:
        # width is 4..7 — no AVX block, just scalar
        for k in range(width):
            lines.append(f"    result += a[{k}] * b[{k}];")

    # NEON path (preferred on arm64)
    lines.append(f"#elif defined(__ARM_NEON) || defined(__ARM_NEON__)")
    if n_neon > 0 or n_avx > 0:
        # Total NEON blocks: cover the whole width in 4-float chunks
        total_neon = width // 4
        rem = width % 4
        lines.append(f"    {{")
        lines.append(f"        float32x4_t acc = vdupq_n_f32(0.0f);")
        for k in range(total_neon):
            lines.append(f"        float32x4_t va_{k} = vld1q_f32(a + {k*4});")
            lines.append(f"        float32x4_t vb_{k} = vld1q_f32(b + {k*4});")
            lines.append(f"        acc = vfmaq_f32(acc, va_{k}, vb_{k});")
        lines.append(f"        float tmp[4]; vst1q_f32(tmp, acc);")
        lines.append(f"        result = tmp[0] + tmp[1] + tmp[2] + tmp[3];")
        offset = total_neon * 4
        for k in range(rem):
            lines.append(f"        result += a[{offset+k}] * b[{offset+k}];")
        lines.append(f"    }}")
    else:
        for k in range(width):
            lines.append(f"    result += a[{k}] * b[{k}];")

    # Scalar fallback
    lines.append(f"#else")
    for k in range(width):
        lines.append(f"    result += a[{k}] * b[{k}];")
    lines.append(f"#endif")

    lines.append(f"    return result;")
    lines.append(f"}}")
    return lines


def _emit_simd_qdot_helper(width: int) -> List[str]:
    """Emit a SIMD dot-product helper: float[N] × int8[N]+scale → float.

    Used when quantize='int8'. The concept is stored as int8 + scale; the query
    is float. The helper dequantizes the int8 vector on the fly and does the dot.
    """
    lines = []
    lines.append(f"/* quantized dot: float[{width}] × int8[{width}]*scale — SIMD dequantize */")
    lines.append(f"static inline float _lal_dot_qsimd_{width}(const float* a, const int8_t* b_q, float b_scale) {{")
    lines.append(f"    float result = 0.0f;")

    n_avx = width // 8
    rem_after_avx = width % 8
    total_neon = width // 4
    rem_neon = width % 4

    # AVX2 path: load int8, widen to int32, convert to float, scale, then fmadd
    lines.append(f"#if defined(__AVX2__) || defined(__AVX__)")
    if n_avx > 0:
        lines.append(f"    {{")
        lines.append(f"        __m256 acc = _mm256_setzero_ps();")
        for k in range(n_avx):
            # Load 8 int8, widen to 8 int32, convert to float
            lines.append(f"        __m128i bi8_{k} = _mm_loadl_epi64((const __m128i*)(b_q + {k*8}));")
            lines.append(f"        __m256i bi32_{k} = _mm256_cvtepi8_epi32(bi8_{k});")
            lines.append(f"        __m256 bf_{k} = _mm256_cvtepi32_ps(bi32_{k});")
            lines.append(f"        __m256 bs_{k} = _mm256_mul_ps(bf_{k}, _mm256_set1_ps(b_scale));")
            lines.append(f"        __m256 va_{k} = _mm256_loadu_ps(a + {k*8});")
            lines.append(f"        acc = _mm256_fmadd_ps(va_{k}, bs_{k}, acc);")
        lines.append(f"        __m128 lo = _mm256_castps256_ps128(acc);")
        lines.append(f"        __m128 hi = _mm256_extractf128_ps(acc, 1);")
        lines.append(f"        __m128 s = _mm_add_ps(lo, hi);")
        lines.append(f"        s = _mm_hadd_ps(s, s);")
        lines.append(f"        s = _mm_hadd_ps(s, s);")
        lines.append(f"        result = _mm_cvtss_f32(s);")
        offset = n_avx * 8
        for k in range(rem_after_avx):
            lines.append(f"        result += a[{offset+k}] * ((float)b_q[{offset+k}] * b_scale);")
        lines.append(f"    }}")
    else:
        for k in range(width):
            lines.append(f"    result += a[{k}] * ((float)b_q[{k}] * b_scale);")

    # NEON path
    lines.append(f"#elif defined(__ARM_NEON) || defined(__ARM_NEON__)")
    if total_neon > 0:
        lines.append(f"    {{")
        lines.append(f"        float32x4_t acc = vdupq_n_f32(0.0f);")
        for k in range(total_neon):
            # Load 4 int8, widen to int32, convert to float
            lines.append(f"        int8x8_t bi8_{k} = vld1_s8(b_q + {k*4});")
            lines.append(f"        int32x4_t bi32_{k} = vmovl_s16(vget_low_s16(vmovl_s8(bi8_{k})));")
            lines.append(f"        float32x4_t bf_{k} = vcvtq_f32_s32(bi32_{k});")
            lines.append(f"        float32x4_t bs_{k} = vmulq_n_f32(bf_{k}, b_scale);")
            lines.append(f"        float32x4_t va_{k} = vld1q_f32(a + {k*4});")
            lines.append(f"        acc = vfmaq_f32(acc, va_{k}, bs_{k});")
        lines.append(f"        float tmp[4]; vst1q_f32(tmp, acc);")
        lines.append(f"        result = tmp[0] + tmp[1] + tmp[2] + tmp[3];")
        offset = total_neon * 4
        for k in range(rem_neon):
            lines.append(f"        result += a[{offset+k}] * ((float)b_q[{offset+k}] * b_scale);")
        lines.append(f"    }}")
    else:
        for k in range(width):
            lines.append(f"    result += a[{k}] * ((float)b_q[{k}] * b_scale);")

    # Scalar fallback
    lines.append(f"#else")
    for k in range(width):
        lines.append(f"    result += a[{k}] * ((float)b_q[{k}] * b_scale);")
    lines.append(f"#endif")

    lines.append(f"    return result;")
    lines.append(f"}}")
    return lines


def _emit_simd_q4dot_helper(width: int) -> List[str]:
    """Emit a SIMD dot-product helper: float[N] × int4[N](packed)+scale → float.

    Used when quantize='int4'. The concept is stored as packed nibbles (2 per byte)
    + scale. The helper unpacks each nibble, converts to float (subtracting 8 to
    get signed), scales, and does the dot.

    Packing: byte[i] = (val[2i+1] + 8) << 4 | (val[2i] + 8).
    Unpack: low = (byte & 0xF) - 8; high = (byte >> 4) - 8.

    v0.6: SIMD nibble unpack on AVX2 (8 nibbles = 4 bytes per block) and NEON.
    Scalar fallback for unsupported targets.
    """
    n_packed = (width + 1) // 2  # bytes needed
    lines = []
    lines.append(f"/* quantized dot: float[{width}] × int4[{width}](packed,{n_packed}B)+scale */")
    lines.append(f"static inline float _lal_dot_q4simd_{width}(const float* a, const uint8_t* b_q4, float b_scale) {{")
    lines.append(f"    float result = 0.0f;")

    # We process 8 nibbles (4 bytes) at a time on AVX2, 8 nibbles on NEON.
    n_simd_blocks = width // 8
    rem = width % 8

    # AVX2 path: 8 nibbles per block. Load 4 bytes, separate into low/high nibbles,
    # interleave, subtract 8, widen to int32, convert to float, fmadd.
    lines.append(f"#if defined(__AVX2__) || defined(__AVX__)")
    if n_simd_blocks > 0:
        lines.append(f"    {{")
        lines.append(f"        __m256 acc = _mm256_setzero_ps();")
        lines.append(f"        const __m128i mask_lo = _mm_set1_epi8(0x0F);")
        for k in range(n_simd_blocks):
            byte_off = k * 4
            # Load 4 bytes (8 nibbles) into a 128-bit register.
            lines.append(f"        __m128i bytes_{k} = _mm_cvtsi32_si128(*(const int*)(b_q4 + {byte_off}));")
            # Low nibbles: bytes & 0x0F  (8 values, in byte lanes 0,2,4,6,8,10,12,14)
            lines.append(f"        __m128i lo_{k} = _mm_and_si128(bytes_{k}, mask_lo);")
            # High nibbles: (bytes >> 4) & 0x0F
            lines.append(f"        __m128i hi_{k} = _mm_and_si128(_mm_srli_epi16(bytes_{k}, 4), mask_lo);")
            # Interleave: we want [lo0, hi0, lo1, hi1, ...] in byte order.
            # unpacklo_epi8(lo, hi) gives [lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3, ...]
            lines.append(f"        __m128i inter_{k} = _mm_unpacklo_epi8(lo_{k}, hi_{k});")
            # Now inter_{k} has 8 nibble values (0..15) in its low 8 bytes.
            # Widen to 8 int32 (only low 8 bytes matter), subtract 8, convert to float.
            lines.append(f"        __m256i i32_{k} = _mm256_cvtepi8_epi32(inter_{k});")
            lines.append(f"        __m256i sub_{k} = _mm256_sub_epi32(i32_{k}, _mm256_set1_epi32(8));")
            lines.append(f"        __m256 bf_{k} = _mm256_cvtepi32_ps(sub_{k});")
            lines.append(f"        __m256 bs_{k} = _mm256_mul_ps(bf_{k}, _mm256_set1_ps(b_scale));")
            lines.append(f"        __m256 va_{k} = _mm256_loadu_ps(a + {k*8});")
            lines.append(f"        acc = _mm256_fmadd_ps(va_{k}, bs_{k}, acc);")
        lines.append(f"        __m128 lo = _mm256_castps256_ps128(acc);")
        lines.append(f"        __m128 hi = _mm256_extractf128_ps(acc, 1);")
        lines.append(f"        __m128 s = _mm_add_ps(lo, hi);")
        lines.append(f"        s = _mm_hadd_ps(s, s);")
        lines.append(f"        s = _mm_hadd_ps(s, s);")
        lines.append(f"        result = _mm_cvtss_f32(s);")
        # Scalar tail
        offset = n_simd_blocks * 8
        for i in range(rem):
            byte_idx = i // 2
            nibble_off = offset + i
            byte_off_tail = nibble_off // 2
            shift = (nibble_off % 2) * 4
            lines.append(f"        result += a[{nibble_off}] * ((float)(((b_q4[{byte_off_tail}] >> {shift}) & 0xF) - 8) * b_scale);")
        lines.append(f"    }}")
    else:
        # width < 8: scalar
        for i in range(width):
            byte_idx = i // 2
            shift = (i % 2) * 4
            lines.append(f"    result += a[{i}] * ((float)(((b_q4[{byte_idx}] >> {shift}) & 0xF) - 8) * b_scale);")

    # NEON path: 8 nibbles (4 bytes) per block.
    lines.append(f"#elif defined(__ARM_NEON) || defined(__ARM_NEON__)")
    if n_simd_blocks > 0:
        lines.append(f"    {{")
        lines.append(f"        float32x4_t acc0 = vdupq_n_f32(0.0f);")
        lines.append(f"        float32x4_t acc1 = vdupq_n_f32(0.0f);")
        for k in range(n_simd_blocks):
            byte_off = k * 4
            # Load 4 bytes, unpack to 8 uint16, subtract 8, convert to float.
            lines.append(f"        uint8x8_t bytes_{k} = vld1_u8(b_q4 + {byte_off});  /* only low 4 used */")
            lines.append(f"        uint16x8_t ext_{k} = vmovl_u8(bytes_{k});")
            lines.append(f"        uint16x8_t lo_{k} = vandq_u16(ext_{k}, vdupq_n_u16(0x0F));")
            lines.append(f"        uint16x8_t hi_{k} = vshrq_n_u16(vandq_u16(ext_{k}, vdupq_n_u16(0xF0)), 4);")
            # Interleave lo and hi: we want [lo0, hi0, lo1, hi1, ...]
            lines.append(f"        uint16x8_t inter_{k} = vzipq_u16(lo_{k}, hi_{k}).val[0];")
            # Take low 8 uint16, subtract 8, convert to int32 then float
            lines.append(f"        int16x8_t s16_{k} = vsubq_s16(vreinterpretq_s16_u16(inter_{k}), vdupq_n_s16(8));")
            lines.append(f"        int32x4_t low_{k} = vmovl_s16(vget_low_s16(s16_{k}));")
            lines.append(f"        int32x4_t high_{k} = vmovl_s16(vget_high_s16(s16_{k}));")
            lines.append(f"        float32x4_t f0_{k} = vcvtq_f32_s32(low_{k});")
            lines.append(f"        float32x4_t f1_{k} = vcvtq_f32_s32(high_{k});")
            lines.append(f"        float32x4_t s0_{k} = vmulq_n_f32(f0_{k}, b_scale);")
            lines.append(f"        float32x4_t s1_{k} = vmulq_n_f32(f1_{k}, b_scale);")
            lines.append(f"        float32x4_t va0_{k} = vld1q_f32(a + {k*8});")
            lines.append(f"        float32x4_t va1_{k} = vld1q_f32(a + {k*8+4});")
            lines.append(f"        acc0 = vfmaq_f32(acc0, va0_{k}, s0_{k});")
            lines.append(f"        acc1 = vfmaq_f32(acc1, va1_{k}, s1_{k});")
        lines.append(f"        float32x4_t acc = vaddq_f32(acc0, acc1);")
        lines.append(f"        float tmp[4]; vst1q_f32(tmp, acc);")
        lines.append(f"        result = tmp[0] + tmp[1] + tmp[2] + tmp[3];")
        # Scalar tail
        offset = n_simd_blocks * 8
        for i in range(rem):
            nibble_off = offset + i
            byte_off_tail = nibble_off // 2
            shift = (nibble_off % 2) * 4
            lines.append(f"        result += a[{nibble_off}] * ((float)(((b_q4[{byte_off_tail}] >> {shift}) & 0xF) - 8) * b_scale);")
        lines.append(f"    }}")
    else:
        for i in range(width):
            byte_idx = i // 2
            shift = (i % 2) * 4
            lines.append(f"    result += a[{i}] * ((float)(((b_q4[{byte_idx}] >> {shift}) & 0xF) - 8) * b_scale);")

    # Scalar fallback
    lines.append(f"#else")
    for i in range(width):
        byte_idx = i // 2
        shift = (i % 2) * 4
        lines.append(f"    result += a[{i}] * ((float)(((b_q4[{byte_idx}] >> {shift}) & 0xF) - 8) * b_scale);")
    lines.append(f"#endif")

    lines.append(f"    return result;")
    lines.append(f"}}")
    return lines


def _quantize_int8(vec: List[float]) -> Tuple[List[int], float]:
    """Quantize a float vector to int8 with a per-vector scale.
    Returns (int8_values, scale) where dequantized[i] = int8[i] * scale.
    """
    max_abs = max(abs(v) for v in vec) if vec else 0.0
    if max_abs < 1e-9:
        return [0] * len(vec), 0.0
    scale = max_abs / 127.0
    q = [int(round(v / scale)) for v in vec]
    # Clamp to int8 range
    q = [max(-128, min(127, x)) for x in q]
    return q, scale


def _quantize_int4(vec: List[float]) -> Tuple[List[int], float]:
    """Quantize a float vector to int4 with a per-vector scale.
    Returns (int4_values_in_range_-8..7, scale) where dequantized[i] = int4[i] * scale.
    int4 range: -8 to 7 (8 distinct negative values, 8 non-negative).
    """
    max_abs = max(abs(v) for v in vec) if vec else 0.0
    if max_abs < 1e-9:
        return [0] * len(vec), 0.0
    scale = max_abs / 7.0  # use 7 as max to match int4 positive range
    q = [int(round(v / scale)) for v in vec]
    # Clamp to int4 range
    q = [max(-8, min(7, x)) for x in q]
    return q, scale


def _pack_nibbles(int4_values: List[int]) -> List[int]:
    """Pack a list of int4 values (-8..7) into bytes, two per byte.
    Low nibble = first value, high nibble = second value.
    Returns a list of byte values (0..255).
    """
    # Convert to unsigned 4-bit (0..15) by adding 8: -8→0, 0→8, 7→15
    unsigned = [x + 8 for x in int4_values]
    # Pad to even length
    if len(unsigned) % 2 != 0:
        unsigned.append(8)  # pad with 0 (which is -8 in signed)
    packed = []
    for i in range(0, len(unsigned), 2):
        byte = (unsigned[i+1] << 4) | unsigned[i]
        packed.append(byte)
    return packed


def _emit_rule(rule, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rules_to_emit, max_recursion_depth, name_suffix=""):
    """Emit one rule as a C function. name_suffix is used for multi-rule variants."""
    vec_dim = len(next(iter(concept_map.values()))) if concept_map else 0

    # Determine output signature: each output is either a float[N] (vector) or an int* (argmax index).
    out_params = []
    out_decls = []
    for out_name in rule.outputs:
        matching = [s for s in rule.body if s.var == out_name]
        if matching and isinstance(matching[0].expr, EArgMax):
            out_params.append(f"int* out_{out_name}_idx")
        else:
            out_params.append(f"float* out_{out_name}")
    param_sig = ", ".join([f"const float* {p}" for p in rule.params] + out_params)

    lines = []
    rule_c_name = f"rule_{rule.name}{name_suffix}"
    lines.append(f"/* rule {rule.name}{name_suffix} */")
    if rule.guard is not None:
        lines.append(f"/* guard: {rule.guard} */")
    lines.append(f"void {rule_c_name}({param_sig}) {{")

    # Find which (param, bound) pairs are used in this rule's body AND guard.
    param_bound_pairs = set()
    for stmt in rule.body:
        _collect_param_bound_pairs(stmt.expr, relate_map, concept_names, rule.params, param_bound_pairs)
    if rule.guard is not None:
        _collect_param_bound_pairs(rule.guard, relate_map, concept_names, rule.params, param_bound_pairs)
    for (pname, bname) in sorted(param_bound_pairs):
        dims = bound_map[bname]
        idx_str = ", ".join(f"{pname}[{d}]" for d in dims)
        lines.append(f"    float {pname}__{bname}[{len(dims)}] = {{{idx_str}}};")

    # v0.5: If the rule has a guard, check it first. If falsy, copy the input
    # to the first output and return (base case for recursion).
    if rule.guard is not None:
        var_types_for_guard: Dict[str, str] = {}
        _, guard_c_expr = _emit_scalar_expr(rule.guard, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule.params, var_types_for_guard, max_recursion_depth)
        lines.append(f"    if (!({guard_c_expr})) {{")
        lines.append(f"        /* guard failed: passthrough input as output (base case) */")
        if rule.outputs:
            first_out = rule.outputs[0]
            matching = [s for s in rule.body if s.var == first_out]
            if matching and isinstance(matching[0].expr, EArgMax):
                lines.append(f"        *out_{first_out}_idx = 0;  /* argmax default */")
            else:
                pname = rule.params[0]
                lines.append(f"        for (int i = 0; i < {vec_dim}; i++) out_{first_out}[i] = {pname}[i];")
        lines.append(f"        return;")
        lines.append(f"    }}")

    # Emit body statements.
    # Vars can be: float (scalar), float[N] (vector), or int (argmax index).
    # We track the type of each var.
    var_types: Dict[str, str] = {}  # "scalar" | "vector" | "int"
    for stmt in rule.body:
        if isinstance(stmt.expr, EArgMax):
            _emit_argmax(lines, stmt.var, stmt.expr, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule.params, var_types, max_recursion_depth, vec_dim)
            var_types[stmt.var] = "int"
            continue
        # v0.6: Check if a bare EVar refers to a vector (rule param or known vector var).
        # If so, treat the assignment as a vector copy.
        is_vec_var = (isinstance(stmt.expr, EVar)
                      and (stmt.expr.name in rule.params or var_types.get(stmt.expr.name) == "vector"
                           or stmt.expr.name in concept_names))
        is_vec = _expr_is_vector(stmt.expr, relate_map, rule_map) or is_vec_var
        if is_vec:
            # v0.7: linear layers may have out_dim != vec_dim. Determine the
            # output dimension and declare the array accordingly.
            out_dim = vec_dim
            if isinstance(stmt.expr, ERelateCall):
                r = relate_map[stmt.expr.name]
                if r.op == "linear" and r.weight_data:
                    out_dim = len(r.weight_data)
            lines.append(f"    float {stmt.var}[{out_dim}] = {{0}};")
            _emit_vector_expr(lines, stmt.var, stmt.expr, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule.params, var_types, max_recursion_depth, vec_dim)
            var_types[stmt.var] = "vector"
        else:
            ctype, cexpr = _emit_scalar_expr(stmt.expr, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule.params, var_types, max_recursion_depth)
            lines.append(f"    {ctype} {stmt.var} = {cexpr};")
            var_types[stmt.var] = "scalar"

    # Write outputs.
    lines.append("")
    lines.append("    /* Outputs */")
    for out_name in rule.outputs:
        matching = [s for s in rule.body if s.var == out_name]
        if matching and isinstance(matching[0].expr, EArgMax):
            # argmax already wrote to out_<out_name>_idx via _emit_argmax
            pass
        else:
            # Vector output: copy from local var to out_<out_name>
            lines.append(f"    for (int i = 0; i < {vec_dim}; i++) out_{out_name}[i] = {out_name}[i];")
    lines.append("}")
    return lines


def _expr_is_vector(expr, relate_map, rule_map):
    """Does this expression produce a vector (vs a scalar)?"""
    if isinstance(expr, ERelateCall):
        r = relate_map[expr.name]
        return r.op in ("bind", "bundle", "permute", "vadd", "vsub", "linear")
    if isinstance(expr, ERuleCall):
        return True  # rule calls return vectors
    if isinstance(expr, ERecall):
        return True
    if isinstance(expr, ECall):
        if e_func_name(expr) == "if":
            # ternary: vector if either branch is a vector
            t = _expr_is_vector(expr.args[1], relate_map, rule_map)
            f = _expr_is_vector(expr.args[2], relate_map, rule_map)
            return t or f
        # arithmetic on scalars stays scalar
        return False
    if isinstance(expr, EVar):
        # A bare variable is a vector if it's a rule param or a known vector var.
        # We check this in _emit_rule by passing var_types; here we conservatively
        # return True and let _emit_rule handle it. But to avoid false positives
        # for scalar vars, we return False — _emit_rule special-cases param/var
        # assignment separately.
        return False
    return False

def e_func_name(call_expr):
    """Helper: get the function name from an ECall (or None)."""
    if isinstance(call_expr, ECall):
        return call_expr.func
    return None


def _emit_scalar_expr(e, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth):
    """Emit a scalar expression. Returns (C_type, C_expression)."""
    if isinstance(e, EFloat):
        return "float", f"{e.val:.6f}f"
    if isinstance(e, EVar):
        return "float", e.name
    if isinstance(e, ECall):
        if e.func == "dot":
            raise RuntimeError("bare dot — must be inside a relate")
        # 'if' ternary: scalar case → emit C ternary (short-circuits in C too)
        if e.func == "if":
            cond_str = _emit_scalar_expr(e.args[0], concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth)[1]
            true_str = _emit_scalar_expr(e.args[1], concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth)[1]
            false_str = _emit_scalar_expr(e.args[2], concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth)[1]
            return "float", f"(({cond_str}) ? ({true_str}) : ({false_str}))"
        arg_strs = [_emit_scalar_expr(a, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth)[1] for a in e.args]
        if e.func == "and":
            return "float", f"({arg_strs[0]} * {arg_strs[1]})"
        if e.func == "or":
            return "float", f"({arg_strs[0]} + {arg_strs[1]} - ({arg_strs[0]}) * ({arg_strs[1]}))"
        if e.func == "not":
            return "float", f"(1.0f - ({arg_strs[0]}))"
        if e.func in ("lt", "gt", "le", "ge"):
            op = {"lt": "<", "gt": ">", "le": "<=", "ge": ">="}[e.func]
            return "float", f"(({arg_strs[0]} {op} {arg_strs[1]}) ? 1.0f : 0.0f)"
        if e.func == "add":
            return "float", f"({arg_strs[0]} + {arg_strs[1]})"
        if e.func == "sub":
            return "float", f"({arg_strs[0]} - {arg_strs[1]})"
        if e.func == "mul":
            return "float", f"({arg_strs[0]} * {arg_strs[1]})"
        raise RuntimeError(f"unknown func: {e.func}")
    if isinstance(e, ERelateCall):
        r = relate_map[e.name]
        if r.op == "dot":
            # Find the concept and the var
            a0, a1 = e.args[0], e.args[1]
            if isinstance(a1, EVar) and a1.name in concept_names:
                var, concept_name = a0, a1.name
            elif isinstance(a0, EVar) and a0.name in concept_names:
                var, concept_name = a1, a0.name
            else:
                raise RuntimeError("relate dot must have one concept ref and one input var")
            if not isinstance(var, EVar):
                raise RuntimeError("relate dot's non-concept arg must be a variable")
            var_name = var.name
            is_param = var_name in rule_params
            is_local_vec = var_types.get(var_name) == "vector"
            if r.bound_name:
                dims = bound_map[r.bound_name]
                width = len(dims)
                # Helper to build the scalar dequant term for element k of a masked array.
                def _masked_q_term(a_elem, k):
                    if _QUANTIZE_MODE == "int8":
                        return f"{a_elem} * ((float){concept_name}__{r.bound_name}_q[{k}] * {concept_name}__{r.bound_name}_scale)"
                    elif _QUANTIZE_MODE == "int4":
                        byte_idx = k // 2
                        nibble = f"({concept_name}__{r.bound_name}_q4[{byte_idx}] >> {(k%2)*4}) & 0xF"
                        return f"{a_elem} * ((float)({nibble} - 8) * {concept_name}__{r.bound_name}_scale)"
                    else:
                        return f"{a_elem} * {concept_name}__{r.bound_name}[{k}]"
                if width >= 8:
                    if is_param:
                        a_expr = f"{var_name}__{r.bound_name}"
                    elif is_local_vec:
                        # Local vector indexed at bound dims — fall back to scalar
                        terms = [_masked_q_term(f"{var_name}[{d}]", k) for k, d in enumerate(dims)]
                        return "float", "(" + " + ".join(terms) + ")"
                    else:
                        raise RuntimeError(f"relate dot var {var_name} is neither param nor local vector")
                    _SIMD_HELPERS_NEEDED.add(width)
                    if _QUANTIZE_MODE == "int8":
                        return "float", f"_lal_dot_qsimd_{width}({a_expr}, {concept_name}__{r.bound_name}_q, {concept_name}__{r.bound_name}_scale)"
                    elif _QUANTIZE_MODE == "int4":
                        return "float", f"_lal_dot_q4simd_{width}({a_expr}, {concept_name}__{r.bound_name}_q4, {concept_name}__{r.bound_name}_scale)"
                    else:
                        b_expr = f"{concept_name}__{r.bound_name}"
                        return "float", f"_lal_dot_simd_{width}({a_expr}, {b_expr})"
                else:
                    if is_param:
                        terms = [_masked_q_term(f"{var_name}__{r.bound_name}[{k}]", k) for k, d in enumerate(dims)]
                    elif is_local_vec:
                        terms = [_masked_q_term(f"{var_name}[{d}]", k) for k, d in enumerate(dims)]
                    else:
                        raise RuntimeError(f"relate dot var {var_name} is neither param nor local vector")
                    return "float", "(" + " + ".join(terms) + ")"
            else:
                # Full-dim dot
                width = len(concept_map[concept_name])
                def _full_q_term(d):
                    if _QUANTIZE_MODE == "int8":
                        return f"{var_name}[{d}] * ((float)concept_{concept_name}_q[{d}] * concept_{concept_name}_scale)"
                    elif _QUANTIZE_MODE == "int4":
                        byte_idx = d // 2
                        nibble = f"(concept_{concept_name}_q4[{byte_idx}] >> {(d%2)*4}) & 0xF"
                        return f"{var_name}[{d}] * ((float)({nibble} - 8) * concept_{concept_name}_scale)"
                    else:
                        return f"{var_name}[{d}] * concept_{concept_name}[{d}]"
                if width >= 8:
                    _SIMD_HELPERS_NEEDED.add(width)
                    if _QUANTIZE_MODE == "int8":
                        return "float", f"_lal_dot_qsimd_{width}({var_name}, concept_{concept_name}_q, concept_{concept_name}_scale)"
                    elif _QUANTIZE_MODE == "int4":
                        return "float", f"_lal_dot_q4simd_{width}({var_name}, concept_{concept_name}_q4, concept_{concept_name}_scale)"
                    else:
                        return "float", f"_lal_dot_simd_{width}({var_name}, concept_{concept_name})"
                else:
                    terms = [_full_q_term(d) for d in range(width)]
                    return "float", "(" + " + ".join(terms) + ")"
        raise RuntimeError(f"relate op {r.op} is not scalar")
    raise RuntimeError(f"can't emit scalar: {e}")


def _emit_vector_expr(lines, target_var, e, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth, vec_dim):
    """Emit a vector expression, writing into target_var[0..vec_dim-1]."""
    if isinstance(e, ERelateCall):
        r = relate_map[e.name]
        if r.op == "bind":
            # bind(a, b)[i] = a[i] * b[i]
            a, b = e.args[0], e.args[1]
            a_vec = _resolve_vec_arg(a, concept_names, rule_params, var_types)
            b_vec = _resolve_vec_arg(b, concept_names, rule_params, var_types)
            for i in range(vec_dim):
                lines.append(f"    {target_var}[{i}] = {a_vec}[{i}] * {b_vec}[{i}];")
            return
        if r.op == "bundle":
            # bundle(a, b)[i] = (a[i] + b[i]) / max(|a[i] + b[i]|)
            a, b = e.args[0], e.args[1]
            a_vec = _resolve_vec_arg(a, concept_names, rule_params, var_types)
            b_vec = _resolve_vec_arg(b, concept_names, rule_params, var_types)
            for i in range(vec_dim):
                lines.append(f"    {target_var}[{i}] = {a_vec}[{i}] + {b_vec}[{i}];")
            # Normalize: find max abs and divide
            lines.append(f"    float __max_abs = 0.0f;")
            for i in range(vec_dim):
                lines.append(f"    if ({target_var}[{i}] < 0.0f) {{ if (-{target_var}[{i}] > __max_abs) __max_abs = -{target_var}[{i}]; }} else {{ if ({target_var}[{i}] > __max_abs) __max_abs = {target_var}[{i}]; }}")
            lines.append(f"    if (__max_abs > 1e-9f) {{ for (int i = 0; i < {vec_dim}; i++) {target_var}[i] /= __max_abs; }}")
            return
        if r.op == "permute":
            a = e.args[0]
            a_vec = _resolve_vec_arg(a, concept_names, rule_params, var_types)
            k = r.permute_k
            for i in range(vec_dim):
                src = (i - k) % vec_dim
                lines.append(f"    {target_var}[{i}] = {a_vec}[{src}];")
            return
        if r.op == "vadd":
            a, b = e.args[0], e.args[1]
            a_vec = _resolve_vec_arg(a, concept_names, rule_params, var_types)
            b_vec = _resolve_vec_arg(b, concept_names, rule_params, var_types)
            for i in range(vec_dim):
                lines.append(f"    {target_var}[{i}] = {a_vec}[{i}] + {b_vec}[{i}];")
            return
        if r.op == "vsub":
            a, b = e.args[0], e.args[1]
            a_vec = _resolve_vec_arg(a, concept_names, rule_params, var_types)
            b_vec = _resolve_vec_arg(b, concept_names, rule_params, var_types)
            for i in range(vec_dim):
                lines.append(f"    {target_var}[{i}] = {a_vec}[{i}] - {b_vec}[{i}];")
            return
        if r.op == "linear":
            # v0.7: compiled linear layer. Emit the pruned weight as a static
            # sparse array (CSR format: values + indices + row_ptrs), then a
            # loop that only iterates over non-zero entries. This is the core
            # "logic pruning" — weak connections are gone at compile time.
            a = e.args[0]
            a_vec = _resolve_vec_arg(a, concept_names, rule_params, var_types)
            w = r.weight_data  # [out_dim][in_dim], pruned (zeros for dropped)
            bias = r.weight_bias  # [out_dim]
            out_dim = len(w)
            in_dim = len(w[0]) if out_dim > 0 else 0

            # Build CSR representation
            values = []
            col_idx = []
            row_ptr = [0]
            for j in range(out_dim):
                wj = w[j]
                for i in range(in_dim):
                    if wj[i] != 0.0:
                        values.append(wj[i])
                        col_idx.append(i)
                row_ptr.append(len(values))
            nnz = len(values)
            prune_pct = 100.0 * (1.0 - nnz / (out_dim * in_dim)) if out_dim * in_dim > 0 else 0.0

            # Emit static arrays (only once per rule — use a unique prefix)
            arr_prefix = f"_lal_{r.name}_{id(r) & 0xFFFFFF:x}"
            lines.append(f"    /* linear {r.name}: {out_dim} outputs, {nnz}/{out_dim*in_dim} kept ({prune_pct:.1f}% pruned) */")
            # Emit the CSR arrays as static const at function scope
            if nnz > 0:
                vals_str = ", ".join(f"{v:.6f}f" for v in values)
                cols_str = ", ".join(str(c) for c in col_idx)
                rows_str = ", ".join(str(rp) for rp in row_ptr)
                lines.append(f"    static const float {arr_prefix}_vals[{nnz}] = {{{vals_str}}};")
                lines.append(f"    static const int {arr_prefix}_cols[{nnz}] = {{{cols_str}}};")
                lines.append(f"    static const int {arr_prefix}_rowptr[{out_dim+1}] = {{{rows_str}}};")
                # Emit the sparse matmul loop
                lines.append(f"    for (int j = 0; j < {out_dim}; j++) {{")
                lines.append(f"        float s = {bias[j]:.6f}f;")
                lines.append(f"        for (int k = {arr_prefix}_rowptr[j]; k < {arr_prefix}_rowptr[j+1]; k++) {{")
                lines.append(f"            s += {a_vec}[{arr_prefix}_cols[k]] * {arr_prefix}_vals[k];")
                lines.append(f"        }}")
                lines.append(f"        {target_var}[j] = s;")
                lines.append(f"    }}")
            else:
                # Fully pruned — just bias
                for j in range(out_dim):
                    lines.append(f"    {target_var}[{j}] = {bias[j]:.6f}f;  /* fully pruned */")
            return
        raise RuntimeError(f"relate op {r.op} is not vector")
    if isinstance(e, ERuleCall):
        # Call the sub-rule, passing its output into target_var.
        sub_rule = rule_map[e.name]
        arg_vec = _resolve_vec_arg(e.args[0], concept_names, rule_params, var_types)
        # The sub-rule's first output goes into target_var.
        # We pass target_var as the sub-rule's first output param.
        # But sub-rules may have multiple outputs — we just use the first.
        if not sub_rule.outputs:
            raise RuntimeError(f"rule {e.name} has no outputs")
        first_out = sub_rule.outputs[0]
        # Check if first output is argmax (int) or vector
        matching = [s for s in sub_rule.body if s.var == first_out]
        if matching and isinstance(matching[0].expr, EArgMax):
            raise RuntimeError(f"can't use rule {e.name}'s argmax output as a vector")
        # Build a temp int out for any argmax outputs, and pass target_var for the first vector output.
        out_args = []
        for o in sub_rule.outputs:
            if o == first_out:
                out_args.append(target_var)
            else:
                matching = [s for s in sub_rule.body if s.var == o]
                if matching and isinstance(matching[0].expr, EArgMax):
                    lines.append(f"    int __{e.name}_{o}_idx = 0;")
                    out_args.append(f"&__{e.name}_{o}_idx")
                else:
                    lines.append(f"    float __{e.name}_{o}[{vec_dim}] = {{0}};")
                    out_args.append(f"__{e.name}_{o}")
        out_args_str = ", ".join(out_args)
        lines.append(f"    rule_{e.name}({arg_vec}{', ' + out_args_str if out_args_str else ''});")
        return
    if isinstance(e, ERecall):
        mem = memory_map[e.memory_name]
        query_vec = _resolve_vec_arg(e.query, concept_names, rule_params, var_types)
        # Soft lookup: weighted average of value vectors, weighted by max(0, dot(query, key))
        # First compute weights.
        n_entries = len(mem.entries)
        lines.append(f"    float __weights[{n_entries}] = {{0}};")
        lines.append(f"    float __total_weight = 0.0f;")
        for k, (key_name, val_name) in enumerate(mem.entries):
            # weight = max(0, dot(query, concept_key))
            terms = [f"{query_vec}[{d}] * concept_{key_name}[{d}]" for d in range(vec_dim)]
            dot_expr = " + ".join(terms)
            lines.append(f"    __weights[{k}] = ({dot_expr});")
            lines.append(f"    if (__weights[{k}] < 0.0f) __weights[{k}] = 0.0f;")
            lines.append(f"    __total_weight += __weights[{k}];")
        # If total_weight is near zero, leave target as zeros.
        lines.append(f"    if (__total_weight > 1e-9f) {{")
        for d in range(vec_dim):
            terms = [f"__weights[{k}] * concept_{mem.entries[k][1]}[{d}]" for k in range(n_entries)]
            lines.append(f"        {target_var}[{d}] = ({' + '.join(terms)}) / __total_weight;")
        lines.append(f"    }}")
        return
    if isinstance(e, ECall) and e.func == "if":
        # Vector ternary: emit if/else block, writing into target_var.
        cond_str = _emit_scalar_expr(e.args[0], concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth)[1]
        lines.append(f"    if ({cond_str}) {{")
        _emit_vector_expr(lines, target_var, e.args[1], concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth, vec_dim)
        lines.append(f"    }} else {{")
        _emit_vector_expr(lines, target_var, e.args[2], concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth, vec_dim)
        lines.append(f"    }}")
        return
    # Bare vector variable or concept ref: copy into target_var.
    if isinstance(e, EVar):
        src = _resolve_vec_arg(e, concept_names, rule_params, var_types)
        for i in range(vec_dim):
            lines.append(f"    {target_var}[{i}] = {src}[{i}];")
        return
    raise RuntimeError(f"can't emit vector expr: {e}")


def _emit_rule_dispatcher(rule_name, variants, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rules_to_emit, max_recursion_depth, vec_dim):
    """Emit a dispatcher function that tries each variant's guard in order,
    calling the first whose guard is truthy. If none match, passthrough input."""
    r0 = variants[0]
    out_params = []
    for out_name in r0.outputs:
        matching = [s for s in r0.body if s.var == out_name]
        if matching and isinstance(matching[0].expr, EArgMax):
            out_params.append(f"int* out_{out_name}_idx")
        else:
            out_params.append(f"float* out_{out_name}")
    param_sig = ", ".join([f"const float* {p}" for p in r0.params] + out_params)
    lines = []
    lines.append(f"/* dispatcher for rule {rule_name} ({len(variants)} variant(s)) */")
    lines.append(f"void rule_{rule_name}({param_sig}) {{")
    # Build the arg list to pass to variants (params + outputs)
    call_args = ", ".join(list(r0.params) + [f"out_{n}" if not any(isinstance(s, Stmt) and s.var == n and isinstance(s.expr, EArgMax) for s in r0.body) else f"out_{n}_idx" for n in r0.outputs])
    # Actually, build call args more carefully — match each output's type
    call_args_list = list(r0.params)
    for out_name in r0.outputs:
        matching = [s for s in r0.body if s.var == out_name]
        if matching and isinstance(matching[0].expr, EArgMax):
            call_args_list.append(f"out_{out_name}_idx")
        else:
            call_args_list.append(f"out_{out_name}")
    call_args = ", ".join(call_args_list)

    for idx, variant in enumerate(variants):
        if variant.guard is not None:
            # Check the guard; if truthy, call this variant and return.
            var_types_g = {}
            _, guard_c_expr = _emit_scalar_expr(variant.guard, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, variant.params, var_types_g, max_recursion_depth)
            # Need to emit masked arrays for the guard's param-bound pairs first.
            # For simplicity, we emit them inline here.
            param_bound_pairs = set()
            _collect_param_bound_pairs(variant.guard, relate_map, concept_names, variant.params, param_bound_pairs)
            for (pname, bname) in sorted(param_bound_pairs):
                dims = bound_map[bname]
                idx_str = ", ".join(f"{pname}[{d}]" for d in dims)
                lines.append(f"    float {pname}__{bname}[{len(dims)}] = {{{idx_str}}};")
            lines.append(f"    if ({guard_c_expr}) {{")
            lines.append(f"        rule_{rule_name}_v{idx}({call_args});")
            lines.append(f"        return;")
            lines.append(f"    }}")
        else:
            # No guard: always matches. Call and return.
            lines.append(f"    rule_{rule_name}_v{idx}({call_args});")
            lines.append(f"    return;")
    # No variant matched: passthrough input as first output
    lines.append(f"    /* no variant matched: passthrough input */")
    if r0.outputs:
        first_out = r0.outputs[0]
        matching = [s for s in r0.body if s.var == first_out]
        if matching and isinstance(matching[0].expr, EArgMax):
            lines.append(f"    *out_{first_out}_idx = 0;")
        else:
            pname = r0.params[0]
            lines.append(f"    for (int i = 0; i < {vec_dim}; i++) out_{first_out}[i] = {pname}[i];")
    lines.append(f"}}")
    return lines


def _resolve_vec_arg(arg, concept_names, rule_params, var_types):
    """Return the C expression that names a vector (for indexing into)."""
    if isinstance(arg, EVar):
        if arg.name in concept_names:
            return f"concept_{arg.name}"
        if arg.name in rule_params:
            return arg.name  # it's a function param (a const float*)
        return arg.name  # it's a local vector var
    raise RuntimeError(f"vector arg must be a var (concept, param, or local); got {arg}")


def _emit_argmax(lines, out_var, expr, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth, vec_dim):
    """Emit argmax: compute each score, then pick max, write index to out_<out_var>_idx."""
    items = expr.items
    score_vars = []
    for k, (label, e) in enumerate(items):
        sv = f"{out_var}_score_{k}"
        t, s = _emit_scalar_expr(e, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth)
        lines.append(f"    float {sv} = {s};  /* {label} */")
        score_vars.append(sv)
    lines.append(f"    *out_{out_var}_idx = 0;")
    lines.append(f"    float __best = {score_vars[0]};")
    for k in range(1, len(score_vars)):
        lines.append(f"    if ({score_vars[k]} > __best) {{ __best = {score_vars[k]}; *out_{out_var}_idx = {k}; }}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="LAL compiler: .lal → specialized C")
    ap.add_argument("input", help="input .lal file")
    ap.add_argument("rule", help="entry rule name")
    ap.add_argument("output", nargs="?", help="output .c file (default: stdout)")
    ap.add_argument("--quantize", choices=["int8", "int4"], default=None,
                    help="quantize concept vectors to int8 (4x smaller) or int4 (8x smaller)")
    args = ap.parse_args()

    with open(args.input) as f:
        source = f.read()

    concepts, bounds, memories, relates, rules = parse(source, args.input)
    c_code = compile_to_c(concepts, bounds, memories, relates, rules, args.rule, quantize=args.quantize)

    if args.output:
        with open(args.output, "w") as f:
            f.write(c_code)
        print(f"[*] compiled {args.input} rule '{args.rule}' -> {args.output}")
        print(f"    concepts: {len(concepts)}, bounds: {len(bounds)}, memories: {len(memories)}, relates: {len(relates)}, rules: {len(rules)}")
        if args.quantize:
            print(f"    quantization: {args.quantize}")
    else:
        print(c_code)


if __name__ == "__main__":
    main()
