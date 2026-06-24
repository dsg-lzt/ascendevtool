from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import libcst as cst
from libcst import matchers as m


_CUDA_DEVICE_STR_RE = re.compile(r"^cuda(:\d+)?$")


class TorchToNpuTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.changes: int = 0
        self._torch_import_index: Optional[int] = None
        self._torch_npu_already_imported: bool = False

    # ── import torch → add import torch_npu ──────────────────────────
    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
        if not node.body:
            return None
        stmt = node.body[0]
        if isinstance(stmt, cst.Import):
            for alias in stmt.names:
                name = self._full_name(alias)
                if name == "torch" and self._torch_import_index is None:
                    self._torch_import_index = len(self._seen_imports) if hasattr(self, '_seen_imports') else None
        return None

    def visit_Module(self, node: cst.Module) -> Optional[bool]:
        idx = 0
        for stmt in node.body:
            if isinstance(stmt, cst.SimpleStatementLine) and stmt.body:
                inner = stmt.body[0]
                if isinstance(inner, cst.Import):
                    for alias in inner.names:
                        name = self._full_name(alias)
                        if name == "torch" and self._torch_import_index is None:
                            self._torch_import_index = idx
                        elif name == "torch_npu":
                            self._torch_npu_already_imported = True
                elif isinstance(inner, cst.ImportFrom):
                    module_val = self._module_str(getattr(inner, "module", None))
                    if module_val and (module_val == "torch_npu" or module_val.startswith("torch.npu")):
                        self._torch_npu_already_imported = True
            idx += 1
        return None

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if self._torch_import_index is None or self._torch_npu_already_imported:
            return updated_node
        new_import = cst.parse_statement("import torch_npu\n")
        new_body: List[cst.BaseStatement] = list(updated_node.body)
        insert_at = self._torch_import_index + 1
        if insert_at >= len(new_body):
            new_body.append(new_import)
        else:
            new_body.insert(insert_at, new_import)
        self.changes += 1
        return updated_node.with_changes(body=tuple(new_body))

    # ── Attribute: torch.cuda → torch.npu / torch.cuda.X → torch.npu.X ──
    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        if self._is_torch_cuda_direct(original_node):
            self.changes += 1
            return updated_node.with_changes(attr=cst.Name("npu"))
        if self._is_torch_cuda_value(original_node):
            self.changes += 1
            return updated_node.with_changes(
                value=updated_node.value.with_changes(attr=cst.Name("npu"))
            )
        return updated_node

    # ── Call: .cuda() → .npu()  ───────────────────────────────────────
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        if isinstance(original_node.func, cst.Attribute):
            attr = original_node.func
            if attr.attr.value == "cuda":
                self.changes += 1
                return updated_node.with_changes(
                    func=cst.Attribute(value=attr.value, attr=cst.Name("npu"), lpar=attr.lpar, rpar=attr.rpar)
                )
        return updated_node

    # ── SimpleString: "cuda" / "cuda:0" → "npu" / "npu:0" ────────────
    def leave_SimpleString(self, original_node: cst.SimpleString, updated_node: cst.SimpleString) -> cst.BaseExpression:
        val = original_node.evaluated_value
        if isinstance(val, str) and _CUDA_DEVICE_STR_RE.match(val):
            new_val = val.replace("cuda", "npu", 1)
            self.changes += 1
            quote = original_node.quote
            return cst.SimpleString(f'{quote}{new_val}{quote}')
        return updated_node

    # ── ImportFrom: from torch.cuda.amp import X → from torch.npu.amp import X ──
    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.BaseSmallStatement:
        module_val = self._module_str(original_node.module)
        if module_val is None:
            return updated_node
        if module_val.startswith("torch.cuda."):
            rest = module_val[len("torch.cuda."):]
            first_dot = rest.find(".")
            if first_dot > 0:
                rest = rest[:first_dot]
            self.changes += 1
            new_module = cst.Attribute(
                value=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("npu")),
                attr=cst.Name(rest),
            )
            return updated_node.with_changes(module=new_module)
        if module_val == "torch.cuda":
            self.changes += 1
            return updated_node.with_changes(
                module=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("npu"))
            )
        return updated_node

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _is_torch_cuda_direct(node: cst.Attribute) -> bool:
        return (
            node.attr.value == "cuda"
            and isinstance(node.value, cst.Name)
            and node.value.value == "torch"
        )

    @staticmethod
    def _is_torch_cuda_value(node: cst.Attribute) -> bool:
        return (
            isinstance(node.value, cst.Attribute)
            and node.value.attr.value == "cuda"
            and isinstance(node.value.value, cst.Name)
            and node.value.value.value == "torch"
        )

    @staticmethod
    def _full_name(alias: cst.ImportAlias) -> str:
        name = alias.name
        if isinstance(name, cst.Name):
            return name.value
        if isinstance(name, cst.Attribute):
            parts = []
            cur = name
            while isinstance(cur, cst.Attribute):
                parts.append(cur.attr.value)
                cur = cur.value
            if isinstance(cur, cst.Name):
                parts.append(cur.value)
            parts.reverse()
            return ".".join(parts)
        return str(getattr(name, "value", name))

    @staticmethod
    def _module_str(module: Optional[cst.BaseExpression]) -> Optional[str]:
        if module is None:
            return None
        if isinstance(module, cst.Attribute):
            parts = []
            cur = module
            while isinstance(cur, cst.Attribute):
                parts.append(cur.attr.value)
                cur = cur.value
            if isinstance(cur, cst.Name):
                parts.append(cur.value)
            parts.reverse()
            return ".".join(parts)
        if isinstance(module, cst.Name):
            return module.value
        return None


def transform_source(source: str) -> Tuple[str, int]:
    tree = cst.parse_module(source)
    transformer = TorchToNpuTransformer()
    new_tree = tree.visit(transformer)
    return new_tree.code, transformer.changes
