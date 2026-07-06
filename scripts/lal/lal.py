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
    op: str                          # "dot", "bind", "bundle", "permute"
    bound_name: Optional[str]        # for dot only
    permute_k: Optional[int]         # for permute only

@dataclass
class Rule:
    name: str
    params: List[str]
    body: List["Stmt"]
    outputs: List[str]

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
        e = self.parse_or()
        self._skip_ws()
        if self.i < self.n:
            raise ParseError(f"trailing content at pos {self.i}: {self.s[self.i:]!r}")
        return e

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
            e = self.parse_or()
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
                query = self.parse_or()
                self._expect(")")
                return ERecall(mem_name, query)
            # function call?
            if self._peek() == "(":
                self._expect("(")
                args = []
                if self._peek() != ")":
                    args.append(self.parse_or())
                    while self._match(","):
                        args.append(self.parse_or())
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
            expr = self.parse_or()
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

def parse(source: str):
    """Parse LAL source into (concepts, bounds, memories, relates, rules)."""
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

        # concept name = [...]
        m = re.match(r"concept\s+(\w+)\s*=\s*(\[.*\])\s*$", line)
        if m:
            concepts.append(Concept(m.group(1), _parse_float_list(m.group(2))))
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

        # rule name(params):
        m = re.match(r"rule\s+(\w+)\s*\(([^)]*)\)\s*:\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
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

            rules.append(Rule(name, params, body, outputs))
            continue

        raise ParseError(f"can't parse line: {line!r}")

    # Post-process: convert ECall(func=relate_name, ...) to ERelateCall,
    # and ECall(func=rule_name, ...) to ERuleCall.
    relate_names = {r.name for r in relates}
    rule_names = {r.name for r in rules}
    for rule in rules:
        for stmt in rule.body:
            stmt.expr = _rewrite_calls(stmt.expr, relate_names, rule_names)

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
    rule_map = {r.name: r for r in rules}
    rule = rule_map[rule_name]

    env: Dict[str, object] = {rule.params[0]: input_vec}

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
            raise RuntimeError(f"unknown relate op: {r.op}")
        if isinstance(e, ERuleCall):
            # Recursively run the called rule with the arg as input
            arg_val = ev(e.args[0])
            sub_env = run_reference(concepts, bounds, memories, relates, rules, e.name, arg_val)
            # Return the first output of the sub-rule
            sub_rule = rule_map[e.name]
            if sub_rule.outputs:
                return sub_env[sub_rule.outputs[0]]
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

    for stmt in rule.body:
        env[stmt.var] = ev(stmt.expr)

    return env


# =============================================================================
# Compiler — AST → specialized C
# =============================================================================

def compile_to_c(concepts, bounds, memories, relates, rules, rule_name, max_recursion_depth=2) -> str:
    """Compile a rule (and any rules it calls, up to max_recursion_depth) to specialized C."""
    concept_map = {c.name: c.vec for c in concepts}
    bound_map = {b.name: b.dims for b in bounds}
    memory_map = {m.name: m for m in memories}
    relate_map = {r.name: r for r in relates}
    rule_map = {r.name: r for r in rules}
    concept_names = set(concept_map.keys())

    # Find which rules we need to emit (transitively, up to max_recursion_depth).
    rules_to_emit = _collect_called_rules(rule_map[rule_name], rule_map, relate_map, max_recursion_depth)
    rules_to_emit.add(rule_name)

    # Determine the vector dimension (assume all concepts have the same dim).
    vec_dim = len(concepts[0].vec) if concepts else 0

    lines = []
    lines.append("/*")
    lines.append(" * Generated by lalc v0.2 — the LAL compiler.")
    lines.append(" *")
    lines.append(f" * Entry rule: {rule_name}")
    lines.append(" * Vector dim: " + str(vec_dim))
    lines.append(" * Rules emitted (transitive closure up to depth " + str(max_recursion_depth) + "):")
    for rn in sorted(rules_to_emit):
        lines.append(f" *   - {rn}")
    lines.append(" *")
    lines.append(" * Specializations applied at compile time:")
    lines.append(" *   - BOUND: only used dimensions retained in concept vectors")
    lines.append(" *   - DOT: fully unrolled, no loops")
    lines.append(" *   - BIND/BUNDLE/PERMUTE: element-wise, fully unrolled")
    lines.append(" *   - MEMORY recall: compiled to fixed dot products + weighted sum")
    lines.append(" *   - Rule calls: inlined up to max_recursion_depth")
    lines.append(" *   - argmax: flat comparison chain, no dispatch")
    lines.append(" */")
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("#include <string.h>")
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
            arr = ", ".join(f"{v:.6f}f" for v in vals)
            lines.append(f"static const float {cname}__{bname}[{len(dims)}] = {{{arr}}};")
        lines.append("")

    # Forward-declare all rules being emitted.
    lines.append("/* === Forward declarations === */")
    for rn in sorted(rules_to_emit):
        r = rule_map[rn]
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
            lines.append(f"void rule_{rn}({params_sig}, {out_sig});")
        else:
            lines.append(f"void rule_{rn}({params_sig});")
    lines.append("")

    # Emit each rule as a C function.
    for rn in sorted(rules_to_emit):
        lines.extend(_emit_rule(rule_map[rn], concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rules_to_emit, max_recursion_depth))
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


def _collect_called_rules(rule, rule_map, relate_map, max_depth, seen=None, depth=0):
    """Find all rules transitively called from this rule, up to max_depth."""
    if seen is None:
        seen = set()
    if depth >= max_depth:
        return seen
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
    """Collect concept names that need full-dim arrays (used by VSA ops, recall, or unbounded dot)."""
    if isinstance(expr, ERelateCall):
        r = relate_map[expr.name]
        if r.op in ("bind", "bundle", "permute"):
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
        # The query might be a concept
        _collect_concepts_needing_full(expr.query, relate_map, concept_names, needed, memory_map)
        # The memory's keys and values are concepts — they need full arrays.
        # (These are also added at the top level via the memory_map loop, but be safe.)
        if expr.memory_name in memory_map:
            for key_name, val_name in memory_map[expr.memory_name].entries:
                needed.add(key_name)
                needed.add(val_name)


def _emit_rule(rule, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rules_to_emit, max_recursion_depth):
    """Emit one rule as a C function."""
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
    lines.append(f"/* rule {rule.name} */")
    lines.append(f"void rule_{rule.name}({param_sig}) {{")

    # Find which (param, bound) pairs are used in this rule's body, and emit masked arrays.
    param_bound_pairs = set()
    for stmt in rule.body:
        _collect_param_bound_pairs(stmt.expr, relate_map, concept_names, rule.params, param_bound_pairs)
    for (pname, bname) in sorted(param_bound_pairs):
        dims = bound_map[bname]
        idx_str = ", ".join(f"{pname}[{d}]" for d in dims)
        lines.append(f"    float {pname}__{bname}[{len(dims)}] = {{{idx_str}}};")

    # Emit body statements.
    # Vars can be: float (scalar), float[N] (vector), or int (argmax index).
    # We track the type of each var.
    var_types: Dict[str, str] = {}  # "scalar" | "vector" | "int"
    for stmt in rule.body:
        if isinstance(stmt.expr, EArgMax):
            _emit_argmax(lines, stmt.var, stmt.expr, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule.params, var_types, max_recursion_depth, vec_dim)
            var_types[stmt.var] = "int"
            continue
        is_vec = _expr_is_vector(stmt.expr, relate_map, rule_map)
        if is_vec:
            lines.append(f"    float {stmt.var}[{vec_dim}] = {{0}};")
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
        return r.op in ("bind", "bundle", "permute")
    if isinstance(expr, ERuleCall):
        return True  # rule calls return vectors
    if isinstance(expr, ERecall):
        return True
    if isinstance(expr, ECall):
        # arithmetic on scalars stays scalar
        return False
    if isinstance(expr, EFloat) or isinstance(expr, EVar):
        return False
    return False


def _emit_scalar_expr(e, concept_map, bound_map, memory_map, relate_map, rule_map, concept_names, rule_params, var_types, max_recursion_depth):
    """Emit a scalar expression. Returns (C_type, C_expression)."""
    if isinstance(e, EFloat):
        return "float", f"{e.val:.6f}f"
    if isinstance(e, EVar):
        return "float", e.name
    if isinstance(e, ECall):
        if e.func == "dot":
            raise RuntimeError("bare dot — must be inside a relate")
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
            # Determine the C expression for accessing the var's elements.
            # - If the var is a rule param (input), we have a masked array `q__bound[k]`.
            # - If the var is a local vector, we index into the full vector: `var[d]`.
            # - If the var is a concept... shouldn't happen (both args are concepts?).
            is_param = var_name in rule_params
            is_local_vec = var_types.get(var_name) == "vector"
            if r.bound_name:
                dims = bound_map[r.bound_name]
                if is_param:
                    # Param: we have a masked array `q__bound[k]` (emitted at function entry).
                    terms = [f"{var_name}__{r.bound_name}[{k}] * {concept_name}__{r.bound_name}[{k}]" for k, d in enumerate(dims)]
                elif is_local_vec:
                    # Local vector: index into the full vector at the bound dims.
                    terms = [f"{var_name}[{d}] * {concept_name}__{r.bound_name}[{k}]" for k, d in enumerate(dims)]
                else:
                    raise RuntimeError(f"relate dot var {var_name} is neither param nor local vector")
                return "float", "(" + " + ".join(terms) + ")"
            else:
                # Full-dim dot
                terms = [f"{var_name}[{d}] * concept_{concept_name}[{d}]" for d in range(len(concept_map[concept_name]))]
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
            # permute(a, k)[i] = a[(i - k) mod N]
            a = e.args[0]
            a_vec = _resolve_vec_arg(a, concept_names, rule_params, var_types)
            k = r.permute_k
            for i in range(vec_dim):
                src = (i - k) % vec_dim
                lines.append(f"    {target_var}[{i}] = {a_vec}[{src}];")
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
    raise RuntimeError(f"can't emit vector expr: {e}")


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
    if len(sys.argv) < 3:
        print("usage: lal.py <input.lal> <rule_name> [output.c]")
        sys.exit(1)
    src_path = sys.argv[1]
    rule_name = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    with open(src_path) as f:
        source = f.read()

    concepts, bounds, memories, relates, rules = parse(source)
    c_code = compile_to_c(concepts, bounds, memories, relates, rules, rule_name)

    if out_path:
        with open(out_path, "w") as f:
            f.write(c_code)
        print(f"[*] compiled {src_path} rule '{rule_name}' -> {out_path}")
        print(f"    concepts: {len(concepts)}, bounds: {len(bounds)}, memories: {len(memories)}, relates: {len(relates)}, rules: {len(rules)}")
    else:
        print(c_code)


if __name__ == "__main__":
    main()
