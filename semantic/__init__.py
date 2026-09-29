"""semantic：地理语义识别（调用形态 → 地理关系语义）。"""

from .semantics import (
    ALIAS_TABLE,
    ASYMMETRIC_PREDICATES,
    PREDICATE_FAMILY,
    PREDICATE_TABLE,
    RelationSemantics,
    build_scope_string,
    combine_sensitivity,
    default_sensitivity_for,
    infer_spatial_scope,
    infer_temporal_scope,
    op_aliases,
    policy_for_operation,
    resolve_relation_semantics,
    split_subject_object,
    suggest_ops,
)

__all__ = [
    "ALIAS_TABLE",
    "ASYMMETRIC_PREDICATES",
    "PREDICATE_FAMILY",
    "PREDICATE_TABLE",
    "RelationSemantics",
    "build_scope_string",
    "combine_sensitivity",
    "default_sensitivity_for",
    "infer_spatial_scope",
    "infer_temporal_scope",
    "op_aliases",
    "policy_for_operation",
    "resolve_relation_semantics",
    "split_subject_object",
    "suggest_ops",
]