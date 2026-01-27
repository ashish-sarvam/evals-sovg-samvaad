# Copyright Sierra
"""
Analysis tools for tau-bench results.
"""

from .failure_analysis import (
    analyze_failure_type,
    find_failing_turn,
    extract_tool_calls,
    FAILURE_TYPE_LEGEND,
)
from .comparison import (
    compare_results,
    load_results,
    ComparisonStatus,
    TaskComparison,
)
from .llm_failure_analysis import (
    analyze_failure,
    analyze_failure_async,
    analyze_failures_batch,
    analyze_failure_heuristic,
    FailureAnalysisResult,
    FAILURE_LABELS,
)

__all__ = [
    "analyze_failure_type",
    "find_failing_turn",
    "extract_tool_calls",
    "FAILURE_TYPE_LEGEND",
    "compare_results",
    "load_results",
    "ComparisonStatus",
    "TaskComparison",
    # LLM-based analysis
    "analyze_failure",
    "analyze_failure_async",
    "analyze_failures_batch",
    "analyze_failure_heuristic",
    "FailureAnalysisResult",
    "FAILURE_LABELS",
]
