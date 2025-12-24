from enum import StrEnum


class GeneralEvaluationResult(StrEnum):
    """Enum representing possible general evaluation results."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    KEY_MISSING = "key_missing"
    KEY_EXTRA = "key_extra"
    JSON_DECODE_ERROR = "json_decode_error"
    BOTH_KEYS_MISSING = "both_keys_missing"


class SemanticSimilarityEvaluationResult(StrEnum):
    """Enum representing possible semantic similarity evaluation results."""

    MATCH = "match"
    PARTIAL_MATCH = "partial_match"
    WRONG = "wrong"


class ToolCallsEvaluationResult(StrEnum):
    """Enum representing possible tool calls evaluation results."""

    TOOL_NAME_CORRECT = "tool_name_correct"
    TOOL_QUERY_CORRECT = "tool_query_correct"
    TOOL_NAME_INCORRECT = "tool_name_incorrect"
    TOOL_QUERY_INCORRECT = "tool_query_incorrect"
    TOOL_NAME_MISSING = "tool_name_missing"
    TOOL_QUERY_MISSING = "tool_query_missing"


class ToolCallsDetailedStatus(StrEnum):
    """Enum representing the overall status of tool calls evaluation."""

    CORRECT = "correct"
    ERROR = "error"


class VariableUpdateStatus(StrEnum):
    """Enum representing the overall status of variable update evaluation."""

    CORRECT = "correct"
    ERROR = "error"


class LineMatchStatus(StrEnum):
    """Enum representing line matching status."""

    FULL_MATCH = "full_match"
    PARTIAL_MATCH = "partial_match"
    NO_MATCH = "no_match"


class LineEvaluationResult:
    """Class representing line evaluation results with match statistics."""

    def __init__(
        self, matched_count: int, total_response_count: int, total_golden_count: int
    ):
        self.matched_count = matched_count
        self.total_response_count = total_response_count
        self.total_golden_count = total_golden_count

    @property
    def response_only_count(self) -> int:
        return self.total_response_count - self.matched_count

    @property
    def golden_only_count(self) -> int:
        return self.total_golden_count - self.matched_count

    @property
    def match_status(self) -> LineMatchStatus:
        # If no lines matched
        if self.matched_count == 0:
            return LineMatchStatus.NO_MATCH

        # If all lines in both sets matched
        if (
            self.matched_count == self.total_response_count
            and self.matched_count == self.total_golden_count
        ):
            return LineMatchStatus.FULL_MATCH

        # Some lines matched but not all
        return LineMatchStatus.PARTIAL_MATCH


class VariableUpdateEvaluationResult:
    """Class representing variable update evaluation results with match statistics."""

    def __init__(
        self,
        correct_keys: int,
        correct_values: int,
        incorrect_keys: int,
        incorrect_values: int,
        total_golden_count: int,
    ):
        self.correct_keys = correct_keys
        self.correct_values = correct_values
        self.incorrect_keys = incorrect_keys
        self.incorrect_values = incorrect_values
        self.total_golden_count = total_golden_count


class ToolCallsDetailedEvaluationResult:
    """Class representing tool calls evaluation results with detailed statistics."""

    def __init__(
        self,
        correct_tool_names: int,
        correct_queries: int,
        incorrect_tool_names: int,
        incorrect_queries: int,
        missing_tool_names: int,
        total_golden_count: int,
        tool_results: dict,
    ):
        self.correct_tool_names = correct_tool_names
        self.correct_queries = correct_queries
        self.incorrect_tool_names = incorrect_tool_names
        self.incorrect_queries = incorrect_queries
        self.missing_tool_names = missing_tool_names
        self.total_golden_count = total_golden_count
        self.tool_results = tool_results
