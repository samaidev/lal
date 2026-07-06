"""
LAL — Logic-Assembly Language
================================

A tiny language for "logic-native compilation". The user writes a logic program
using four primitives:

    concept name = [v0, v1, ...]          # a static vector (a "concept")
    bound   name = [d0, d1, ...]          # a dimension mask (a "boundary")
    relate  name(a, b) = dot(a,b) @ bound # a relation: operator applied within a boundary
    rule    name(query):
        s1 = relate(query, concept1)      # use relations
        s2 = or(s1 > 0.5, s2 > 0.4)       # continuous logic
        output(s2)                        # emit result

The compiler (lalc) turns this into specialized C code:
  - bounds are applied at compile time: unused dims are deleted from vectors
  - dot products are fully unrolled (no loops)
  - logic ops are inlined as arithmetic
  - no generic matmul, no runtime mask indirection, no function call overhead

The output is a standalone C program with no runtime dependencies (just libc).
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union


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
class Relate:
    name: str
    params: List[str]          # e.g. ["a", "b"]
    op: str                    # "dot" (only one for MVP)
    bound_name: Optional[str]  # apply this bound, or None for all dims

@dataclass
class Rule:
    name: str
    params: List[str]          # e.g. ["query"]
    body: List["Stmt"]
    outputs: List[str]         # output var names

@dataclass
class Stmt:
    """A statement in a rule body: var = expr (string form for simplicity)"""
    var: str
    expr: "Expr"

# Expressions: kept as a small algebraic type
@dataclass
class ECall:
    func: str                  # "dot", "and", "or", "not", "lt", "gt", "le", "ge", "const"
    args: List["Expr"]

@dataclass
class EVar:
    name: str

@dataclass
class EConceptRef:
    name: str                  # refers to a Concept by name

@dataclass
class EFloat:
    val: float

@dataclass
class ERelateCall:
    """A call to a declared relation: relate_name(arg0, arg1)"""
    name: str
    args: List["Expr"]

@dataclass
class EArgMax:
    """argmax over a list of (label, expr) pairs — returns the index of the largest"""
    items: List[Tuple[str, "Expr"]]

Expr = Union[ECall, EVar, EConceptRef, EFloat, ERelateCall, EArgMax]


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

def _parse_expr(s: str) -> Expr:
    """Tiny recursive-descent expression parser.

    Grammar:
      expr   := or_expr
      or_expr := and_expr ('or' and_expr)*
      and_expr := cmp_expr ('and' cmp_expr)*
      cmp_expr := atom ('<'|'>'|'<='|'>=') atom
      atom   := NUMBER | NAME | NAME '(' args ')' | argmax '(' pairs ')'
      args   := expr (',' expr)*
    """
    s = s.strip()
    p = _ExprParser(s)
    return p.parse_or()

class _ExprParser:
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
            self.i += len(tok)
            # make sure not part of a longer identifier
            if tok.isalpha() and self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
                self.i -= len(tok)
                return False
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
        left = self.parse_atom()
        for op, name in [("<=", "le"), (">=", "ge"), ("<", "lt"), (">", "gt")]:
            if self._match(op):
                right = self.parse_atom()
                return ECall(name, [left, right])
        return left

    def parse_atom(self) -> Expr:
        self._skip_ws()
        if self.i >= self.n:
            raise ParseError(f"unexpected end of input in: {self.s!r}")

        c = self.s[self.i]
        # parenthesized
        if c == "(":
            self._expect("(")
            e = self.parse_or()
            self._expect(")")
            return e

        # number
        if c.isdigit() or c == "." or (c == "-" and self.i+1 < self.n and (self.s[self.i+1].isdigit() or self.s[self.i+1] == ".")):
            return EFloat(self._read_number())

        # identifier
        if c.isalpha() or c == "_":
            name = self._read_ident()
            # function call?
            if self._peek() == "(":
                self._expect("(")
                args = []
                if self._peek() != ")":
                    args.append(self.parse_or())
                    while self._match(","):
                        args.append(self.parse_or())
                self._expect(")")
                if name == "argmax":
                    # args is a list of (label, value) pairs separated by ';'
                    # We actually want a different syntax: argmax {label: expr, label: expr, ...}
                    # But we already parsed args as expressions. Use a simpler form:
                    #   argmax(cat: e1, dog: e2, car: e3)
                    # For simplicity we accept this special form by re-parsing.
                    raise ParseError("argmax must use {label: expr, ...} form")
                return ECall(name, args)
            return EVar(name)

        raise ParseError(f"unexpected char '{c}' at pos {self.i} in: {self.s!r}")

    # Special parser for argmax {label: expr, ...}
    def parse_argmax(self) -> Expr:
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


def _parse_argmax_special(s: str) -> Expr:
    """Wrapper to parse 'argmax {label: expr, ...}' syntax."""
    s = s.strip()
    # strip leading 'argmax'
    assert s.startswith("argmax"), s
    s = s[len("argmax"):].strip()
    p = _ExprParser(s)
    return p.parse_argmax()


def parse(source: str):
    """Parse LAL source into a list of top-level declarations."""
    concepts = []
    bounds = []
    relates = []
    rules = []

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

        # relate name(params) = dot(a, b) @ bound_name
        m = re.match(r"relate\s+(\w+)\s*\(([^)]*)\)\s*=\s*dot\(([^)]*)\)\s*(?:@\s*(\w+))?\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            dot_args = [a.strip() for a in m.group(3).split(",") if a.strip()]
            bound_name = m.group(4)
            if len(params) != 2 or len(dot_args) != 2:
                raise ParseError(f"relate must have 2 params and dot must have 2 args: {line!r}")
            relates.append(Relate(name, params, "dot", bound_name))
            i += 1
            continue

        # rule name(params):
        m = re.match(r"rule\s+(\w+)\s*\(([^)]*)\)\s*:\s*$", line)
        if m:
            name = m.group(1)
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            body = []
            outputs = []
            i += 1
            # consume indented lines
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
                    # argmax special form — may span multiple lines until } is found
                    if rest.startswith("argmax"):
                        while "}" not in rest:
                            i += 1
                            if i >= len(lines):
                                raise ParseError(f"unterminated argmax starting at: {bline!r}")
                            next_line = lines[i].split("#")[0].rstrip()
                            if next_line.strip():
                                rest += " " + next_line.strip()
                        expr = _parse_argmax_special(rest)
                    else:
                        expr = _parse_expr(rest)
                    body.append(Stmt(var, expr))
                    i += 1
                    continue

                # bare expression as a statement (e.g., argmax without assignment)
                if bline.startswith("argmax"):
                    rest = bline
                    while "}" not in rest:
                        i += 1
                        if i >= len(lines):
                            raise ParseError(f"unterminated argmax starting at: {bline!r}")
                        next_line = lines[i].split("#")[0].rstrip()
                        if next_line.strip():
                            rest += " " + next_line.strip()
                    expr = _parse_argmax_special(rest)
                    body.append(Stmt("__result__", expr))
                    i += 1
                    continue

                raise ParseError(f"can't parse rule body line: {bline!r}")

            rules.append(Rule(name, params, body, outputs))
            continue

        raise ParseError(f"can't parse line: {line!r}")

    # Post-process: convert ECall nodes whose func matches a relate name into ERelateCall.
    relate_names = {r.name for r in relates}
    for rule in rules:
        for stmt in rule.body:
            stmt.expr = _rewrite_relate_calls(stmt.expr, relate_names)

    return concepts, bounds, relates, rules


def _rewrite_relate_calls(expr, relate_names):
    """Walk the expression tree and convert ECall(func=relate_name, ...) to ERelateCall."""
    if isinstance(expr, ECall):
        if expr.func in relate_names:
            new_args = [_rewrite_relate_calls(a, relate_names) for a in expr.args]
            return ERelateCall(expr.func, new_args)
        else:
            new_args = [_rewrite_relate_calls(a, relate_names) for a in expr.args]
            return ECall(expr.func, new_args)
    if isinstance(expr, EArgMax):
        new_items = [(label, _rewrite_relate_calls(e, relate_names)) for label, e in expr.items]
        return EArgMax(new_items)
    return expr


# =============================================================================
# Reference interpreter (Python) — for verification
# =============================================================================

def run_reference(concepts, bounds, relates, rules, rule_name, input_vec):
    """Run a rule with the given input vector. Returns a dict of var → value."""
    concept_map = {c.name: c.vec for c in concepts}
    bound_map = {b.name: b.dims for b in bounds}
    relate_map = {r.name: r for r in relates}
    rule = next(r for r in rules if r.name == rule_name)

    env = {rule.params[0]: input_vec}
    concept_names = set(concept_map.keys())

    def ev(e):
        if isinstance(e, EFloat):
            return e.val
        if isinstance(e, EVar):
            if e.name in env:
                return env[e.name]
            if e.name in concept_map:
                return concept_map[e.name]
            raise RuntimeError(f"unknown var: {e.name}")
        if isinstance(e, EConceptRef):
            return concept_map[e.name]
        if isinstance(e, ECall):
            if e.func == "dot":
                raise RuntimeError("bare dot in expression — use a relate instead")
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
            raise RuntimeError(f"unknown func: {e.func}")
        if isinstance(e, ERelateCall):
            r = relate_map[e.name]
            arg_vals = [ev(a) for a in e.args]
            a, b = arg_vals[0], arg_vals[1]
            if r.bound_name:
                dims = bound_map[r.bound_name]
                return sum(a[d] * b[d] for d in dims)
            else:
                return sum(x*y for x, y in zip(a, b))
        if isinstance(e, EArgMax):
            scores = [(label, ev(expr)) for label, expr in e.items]
            best = max(scores, key=lambda x: x[1])
            return best[0]  # return label
        raise RuntimeError(f"can't eval: {e}")

    for stmt in rule.body:
        env[stmt.var] = ev(stmt.expr)

    return env


# =============================================================================
# Compiler — AST → specialized C
# =============================================================================

def compile_to_c(concepts, bounds, relates, rules, rule_name) -> str:
    """Compile the given rule to a specialized C program.

    Specializations performed:
      1. BOUND applied at compile time: only the dims in the bound are kept
         in concept vectors and the input.
      2. DOT products fully unrolled (no loops).
      3. AND/OR/NOT/comparisons inlined as arithmetic.
      4. argmax turned into a flat sequence of comparisons.
      5. No function call overhead — everything inlined into one function.
    """
    concept_map = {c.name: c.vec for c in concepts}
    bound_map = {b.name: b.dims for b in bounds}
    relate_map = {r.name: r for r in relates}
    rule = next(r for r in rules if r.name == rule_name)

    # Determine which bounds are actually used in the rule body.
    # For each concept referenced, we need its (possibly masked) values per bound used.
    used_bounds = set()
    for stmt in rule.body:
        _collect_used_bounds(stmt.expr, relate_map, used_bounds)

    # For each (concept, bound) pair referenced, emit a masked array.
    # Also for the input variable, emit a masked array per bound used.
    input_var = rule.params[0]
    input_dim = len(concepts[0].vec) if concepts else 0

    # Collect concept refs in the rule body
    concept_names = set(concept_map.keys())
    used_concepts = set()
    for stmt in rule.body:
        _collect_concept_refs(stmt.expr, used_concepts, concept_names)

    # Collect (concept, bound) pairs that are actually used together in a relate call.
    # This is more precise than the cross product — we only emit arrays we'll actually read.
    used_concept_bound_pairs = set()  # (concept_name, bound_name)
    def _collect_pairs(expr):
        if isinstance(expr, ERelateCall):
            r = relate_map[expr.name]
            if r.bound_name:
                # Find the concept arg
                for a in expr.args:
                    if isinstance(a, EVar) and a.name in concept_names:
                        used_concept_bound_pairs.add((a.name, r.bound_name))
        if isinstance(expr, ECall):
            for a in expr.args:
                _collect_pairs(a)
        if isinstance(expr, EArgMax):
            for _, e in expr.items:
                _collect_pairs(e)
    for stmt in rule.body:
        _collect_pairs(stmt.expr)

    lines = []
    lines.append("/*")
    lines.append(" * Generated by lalc — the LAL compiler.")
    lines.append(" *")
    lines.append(" * Source rule: " + rule_name)
    lines.append(" * Specializations applied at compile time:")
    lines.append(" *   - BOUND: only used dimensions retained in concept vectors")
    lines.append(" *   - DOT: fully unrolled, no loops")
    lines.append(" *   - AND/OR/comparisons: inlined as arithmetic")
    lines.append(" *   - argmax: flat comparison chain, no dispatch")
    lines.append(" *   - No generic matmul, no runtime mask indirection")
    lines.append(" */")
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("#include <string.h>")
    lines.append("")

    # Emit masked concept arrays — ONLY for (concept, bound) pairs that are actually used.
    lines.append("/* === Concept vectors (after BOUND applied at compile time) ===")
    lines.append(" * Only (concept, bound) pairs actually referenced in the rule body are emitted.")
    lines.append(" * Unused combinations are eliminated at compile time. */")
    for (cname, bname) in sorted(used_concept_bound_pairs):
        cvec = concept_map[cname]
        dims = bound_map[bname]
        vals = [cvec[d] for d in dims]
        arr = ", ".join(f"{v:.6f}f" for v in vals)
        lines.append(f"static const float {cname}__{bname}[{len(dims)}] = {{{arr}}};")
    lines.append("")

    # The C function takes the input vector (full dim) and applies bounds internally.
    lines.append(f"/* === Compiled rule: {rule_name} === */")
    lines.append(f"/* Input: float q[{input_dim}] (full-dim query vector) */")
    out_names = rule.outputs if rule.outputs else ["__result__"]
    out_sig = ", ".join(f"int* out_{n}" for n in out_names)
    lines.append(f"void {rule_name}(const float* q, {out_sig}) {{")
    lines.append("    /* Apply BOUNDs to the input at compile-time-known indices */")
    for bname in sorted(used_bounds):
        dims = bound_map[bname]
        idx_str = ", ".join(f"q[{d}]" for d in dims)
        lines.append(f"    float q__{bname}[{len(dims)}] = {{{idx_str}}};")
    lines.append("")

    # Emit body
    for stmt in rule.body:
        if isinstance(stmt.expr, EArgMax):
            # argmax is handled at output time; skip emitting it as a statement
            continue
        ctype, cexpr = _emit_expr(stmt.expr, relate_map, bound_map, concept_names)
        lines.append(f"    {ctype} {stmt.var} = {cexpr};")

    # Output: for argmax, emit inline score computation + max selection.
    lines.append("")
    lines.append("    /* Outputs */")
    for out in out_names:
        # Find the statement that produced this output var
        matching = [s for s in rule.body if s.var == out]
        if matching:
            stmt = matching[0]
            if isinstance(stmt.expr, EArgMax):
                _emit_argmax_output(lines, out, stmt.expr, relate_map, bound_map, concept_names)
            else:
                lines.append(f"    *out_{out} = (int){out};")
        else:
            lines.append(f"    *out_{out} = 0;  /* unused */")
    lines.append("}")
    lines.append("")

    # main()
    lines.append("int main(int argc, char** argv) {")
    lines.append(f"    float q[{input_dim}];")
    lines.append(f"    if (argc < {input_dim+1}) {{")
    lines.append(f"        fprintf(stderr, \"usage: {argv0(rule_name)} v0 v1 ... v{input_dim-1}\\n\");")
    lines.append("        return 1;")
    lines.append("    }")
    lines.append(f"    for (int i = 0; i < {input_dim}; i++) q[i] = (float)atof(argv[i+1]);")
    out_decls = "".join(f" int out_{n} = -1;" for n in out_names)
    out_args = "".join(f", &out_{n}" for n in out_names)
    lines.append(f"    {out_decls}")
    lines.append(f"    {rule_name}(q{out_args});")
    # Print outputs
    for n in out_names:
        lines.append(f'    printf("%d\\n", out_{n});')
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def argv0(rule_name):
    return "demo"

def _collect_used_bounds(expr, relate_map, used):
    if isinstance(expr, ERelateCall):
        r = relate_map[expr.name]
        if r.bound_name:
            used.add(r.bound_name)
    if isinstance(expr, ECall):
        for a in expr.args:
            _collect_used_bounds(a, relate_map, used)
    if isinstance(expr, EArgMax):
        for _, e in expr.items:
            _collect_used_bounds(e, relate_map, used)

def _collect_concept_refs(expr, refs, concept_names=None):
    """Collect names of concepts referenced. Since the parser produces EVar
    for all identifiers, we accept a set of known concept names to disambiguate."""
    if isinstance(expr, EConceptRef):
        refs.add(expr.name)
        return
    if isinstance(expr, EVar) and concept_names and expr.name in concept_names:
        refs.add(expr.name)
        return
    if isinstance(expr, ECall):
        for a in expr.args:
            _collect_concept_refs(a, refs, concept_names)
    if isinstance(expr, ERelateCall):
        for a in expr.args:
            _collect_concept_refs(a, refs, concept_names)
    if isinstance(expr, EArgMax):
        for _, e in expr.items:
            _collect_concept_refs(e, refs, concept_names)

def _emit_expr(e, relate_map, bound_map, concept_names=None):
    """Return (C_type, C_expression_string) for the given expression."""
    if isinstance(e, EFloat):
        return "float", f"{e.val:.6f}f"
    if isinstance(e, EVar):
        # If this name is a known concept, treat as concept ref — but bare concept
        # refs in expressions aren't supported (must be inside a relate call).
        # Outside of relate calls, this is a normal variable.
        return "float", e.name
    if isinstance(e, EConceptRef):
        raise RuntimeError("bare concept ref in expression — must be inside a relate call")
    if isinstance(e, ECall):
        if e.func == "dot":
            raise RuntimeError("bare dot — must be inside a relate call")
        arg_types = []
        arg_strs = []
        for a in e.args:
            t, s = _emit_expr(a, relate_map, bound_map, concept_names)
            arg_types.append(t)
            arg_strs.append(s)
        if e.func == "and":
            return "float", f"({arg_strs[0]} * {arg_strs[1]})"
        if e.func == "or":
            return "float", f"({arg_strs[0]} + {arg_strs[1]} - ({arg_strs[0]}) * ({arg_strs[1]}))"
        if e.func == "not":
            return "float", f"(1.0f - ({arg_strs[0]}))"
        if e.func in ("lt", "gt", "le", "ge"):
            op = {"lt": "<", "gt": ">", "le": "<=", "ge": ">="}[e.func]
            # Comparison returns 0/1 as float (continuous logic compatible)
            return "float", f"(({arg_strs[0]} {op} {arg_strs[1]}) ? 1.0f : 0.0f)"
        raise RuntimeError(f"unknown func: {e.func}")
    if isinstance(e, ERelateCall):
        r = relate_map[e.name]
        # Args should be EVar — one referring to the input, one to a concept.
        a0 = e.args[0]
        a1 = e.args[1]
        # Determine which is the concept (by name) and which is the input var.
        if isinstance(a1, EVar) and concept_names and a1.name in concept_names:
            var, concept_name = a0, a1.name
        elif isinstance(a0, EVar) and concept_names and a0.name in concept_names:
            var, concept_name = a1, a0.name
        else:
            raise RuntimeError("relate call must have one concept ref and one input var")
        if not isinstance(var, EVar):
            raise RuntimeError("relate call's non-concept arg must be a variable (the input)")
        var_name = var.name
        if r.bound_name:
            dims = bound_map[r.bound_name]
            # Unrolled dot product: sum over dims of q[d] * concept[d]
            # The concept's masked array is {concept}__{bound}, the input's is q__{bound}.
            terms = []
            for k, d in enumerate(dims):
                terms.append(f"q__{r.bound_name}[{k}] * {concept_name}__{r.bound_name}[{k}]")
            return "float", "(" + " + ".join(terms) + ")"
        else:
            # Full-dim dot — but we haven't stored full-dim concept arrays.
            # For MVP, only support bounded dots.
            raise RuntimeError("relate without bound not yet supported")
    if isinstance(e, EArgMax):
        # argmax returns a label (string). For C, we return an int index.
        # The actual argmax computation happens at output time, so we just
        # stash the items. But the AST eval already gave us a label...
        # For the C emitter, we handle argmax specially in the output step.
        # Here, we return a placeholder that won't be used.
        return "int", "/* argmax — handled at output */"

    raise RuntimeError(f"can't emit: {e}")


def _emit_argmax_output(lines, out_var, expr, relate_map, bound_map, concept_names=None):
    """Emit inline argmax: compute each item's score, then pick the max."""
    items = expr.items
    # Emit each score
    score_vars = []
    for k, (label, e) in enumerate(items):
        sv = f"{out_var}_score_{k}"
        t, s = _emit_expr(e, relate_map, bound_map, concept_names)
        lines.append(f"    float {sv} = {s};  /* {label} */")
        score_vars.append(sv)
    # Pick max
    lines.append(f"    *out_{out_var} = 0;")
    lines.append(f"    float __best = {score_vars[0]};")
    for k in range(1, len(score_vars)):
        lines.append(f"    if ({score_vars[k]} > __best) {{ __best = {score_vars[k]}; *out_{out_var} = {k}; }}")


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

    concepts, bounds, relates, rules = parse(source)
    c_code = compile_to_c(concepts, bounds, relates, rules, rule_name)

    if out_path:
        with open(out_path, "w") as f:
            f.write(c_code)
        print(f"[*] compiled {src_path} rule '{rule_name}' -> {out_path}")
        print(f"    concepts: {len(concepts)}, bounds: {len(bounds)}, relates: {len(relates)}")
    else:
        print(c_code)


if __name__ == "__main__":
    main()
