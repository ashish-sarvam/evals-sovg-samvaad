import json
import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("benchmark_viewer")


# Path resolution
def get_results_dir():
    """Get the path to the results directory"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # Navigate to the results directory (../../results)
    results_dir = script_dir.parent.parent / "results"

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        st.error(f"Results directory not found: {results_dir}")
        return None

    return results_dir


def load_benchmark_file(file_path):
    """Load a benchmark JSON file"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading benchmark file: {e}")
        st.error(f"Error loading benchmark file: {e}")
        return None


def load_css():
    """Load custom CSS"""
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found: {css_file}")


def get_available_evaluation_keys(benchmark_data):
    """Extract all evaluation keys used across samples"""
    eval_keys = set()

    if "samples" in benchmark_data:
        for sample_id, sample in benchmark_data["samples"].items():
            for model, results in sample.get("model_results", {}).items():
                eval_results = results.get("evaluation_results", {})
                if isinstance(eval_results, dict):
                    for key in eval_results.keys():
                        eval_keys.add(key)

    return sorted(list(eval_keys))


def filter_samples(benchmark_data, filter_config):
    """Filter samples based on filter configuration"""
    filtered_samples = {}

    for sample_id, sample in benchmark_data["samples"].items():
        model_results = sample.get("model_results", {})

        # Skip if no model results
        if not model_results:
            continue

        # Check filters - initialize result based on combination mode
        # If ANY mode (OR), start with False and any match will make it True
        # If ALL mode (AND), start with True and any non-match will make it False
        filter_mode = filter_config.get("filter_mode", "ANY")
        include_sample = False if filter_mode == "ANY" else True

        # Track if any filter was applied
        filters_applied = False

        # Filter by model failure count
        if filter_config["filter_by_failure_count"]:
            filters_applied = True
            models_failed = sum(
                1
                for model, result in model_results.items()
                if result.get("has_failed", False)
            )

            result_matches = False
            if (
                filter_config["failure_count_operator"] == "equal"
                and models_failed == filter_config["failure_count"]
            ):
                result_matches = True
            elif (
                filter_config["failure_count_operator"] == "greater_than"
                and models_failed >= filter_config["failure_count"]
            ):
                result_matches = True
            elif (
                filter_config["failure_count_operator"] == "less_than"
                and models_failed <= filter_config["failure_count"]
            ):
                result_matches = True

            # Apply based on filter mode
            if filter_mode == "ANY":
                include_sample = include_sample or result_matches
            else:  # ALL mode
                include_sample = include_sample and result_matches

        # Filter by specific model failure
        if filter_config["filter_by_model_failure"]:
            filters_applied = True
            model_name = filter_config["failed_model"]
            result_matches = (
                model_name in model_results
                and model_results[model_name].get("has_failed", False)
                == filter_config["model_failure_value"]
            )

            # Apply based on filter mode
            if filter_mode == "ANY":
                include_sample = include_sample or result_matches
            else:  # ALL mode
                include_sample = include_sample and result_matches

        # Filter by multiple evaluation key filters
        if filter_config.get("filter_by_evaluation_keys", False) and filter_config.get(
            "eval_filters"
        ):
            filters_applied = True

            # Process each evaluation filter
            for eval_filter in filter_config["eval_filters"]:
                model_name = eval_filter["model"]
                eval_key = eval_filter["key"]
                eval_value = eval_filter["value"]
                result_matches = False

                if model_name in model_results:
                    eval_results = model_results[model_name].get(
                        "evaluation_results", {}
                    )

                    if eval_key in eval_results:
                        # String value evaluation
                        if isinstance(eval_results[eval_key], str):
                            if eval_results[eval_key] == eval_value:
                                result_matches = True
                        # Dictionary with status evaluation
                        elif (
                            isinstance(eval_results[eval_key], dict)
                            and "status" in eval_results[eval_key]
                        ):
                            if eval_results[eval_key]["status"] == eval_value:
                                result_matches = True

                # Apply based on filter mode
                if filter_mode == "ANY":
                    include_sample = include_sample or result_matches
                else:  # ALL mode
                    include_sample = include_sample and result_matches

        # If no filters applied, don't include the sample
        if not filters_applied:
            include_sample = False

        # If sample matches filters, add it to results
        if include_sample:
            filtered_samples[sample_id] = sample

    return filtered_samples


def main():
    st.set_page_config(
        page_title="TOTA Benchmark Viewer",
        page_icon="📊",
        layout="wide",
    )

    # Load custom CSS
    load_css()

    st.title("TOTA Benchmark Viewer")

    # Get the results directory
    results_dir = get_results_dir()
    if not results_dir:
        return

    # Get all JSON files in the results directory
    benchmark_files = [f for f in results_dir.glob("*.json") if f.is_file()]

    if not benchmark_files:
        st.warning("No benchmark files found in the results directory.")
        return

    # Sort files by modification time (newest first)
    benchmark_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Convert to list of filenames for the dropdown
    file_names = [f.name for f in benchmark_files]

    # Add a file selector
    selected_file = st.selectbox("Select a benchmark file", file_names, index=0)

    # Get the full path of the selected file
    selected_file_path = results_dir / selected_file

    # Load the selected benchmark file
    benchmark_data = load_benchmark_file(selected_file_path)

    if not benchmark_data:
        return

    # Display summary information
    st.header("Benchmark Summary")
    if "summary" in benchmark_data:
        summary = benchmark_data["summary"]

        # Display basic information
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Timestamp: {summary.get('timestamp', 'N/A')}")
            st.info(f"Total samples: {summary.get('total_samples', 'N/A')}")

        # Display models summary
        if "models_summary" in summary:
            st.subheader("Models Performance")
            models_summary = summary["models_summary"]

            # Create a metrics row for model performance
            cols = st.columns(len(models_summary))
            for i, (model_name, model_data) in enumerate(models_summary.items()):
                success_rate = model_data.get("success_rate", 0)
                failures = model_data.get("total_failures", 0)
                total = model_data.get("total_evaluated", 0)

                with cols[i]:
                    st.metric(
                        label=model_name,
                        value=f"{success_rate:.2f}%",
                        delta=f"Failures: {failures}/{total}",
                    )

    # Get model names for filtering
    model_names = []
    if "models" in benchmark_data:
        model_names = list(benchmark_data["models"].keys())

    # Get evaluation keys for filtering
    eval_keys = get_available_evaluation_keys(benchmark_data)

    # Display samples
    st.header("Samples")

    # Create tabs
    tab1, tab2 = st.tabs(["Browse Samples", "Filter & Search"])

    with tab1:
        if "samples" in benchmark_data:
            samples = benchmark_data["samples"]
            sample_ids = list(samples.keys())

            # Add a sample selector
            sample_idx = st.selectbox(
                "Select a sample",
                range(len(sample_ids)),
                format_func=lambda i: f"Sample {sample_ids[i]}",
            )
            selected_sample_id = sample_ids[sample_idx]
            selected_sample = samples[selected_sample_id]

            # Display the selected sample
            display_sample(selected_sample)

    with tab2:
        if "samples" in benchmark_data:
            samples = benchmark_data["samples"]

            # Filter configuration
            st.subheader("Filter Samples")

            # Initialize session state variables if not already present
            if "eval_filters" not in st.session_state:
                st.session_state.eval_filters = []
            if "filtered_samples" not in st.session_state:
                st.session_state.filtered_samples = {}
            if "filters_applied" not in st.session_state:
                st.session_state.filters_applied = False
            if "filter_config" not in st.session_state:
                st.session_state.filter_config = {}

            # Display filter count and clear button
            filter_header_col1, filter_header_col2 = st.columns([3, 1])

            with filter_header_col1:
                active_filters = len(st.session_state.eval_filters)
                if active_filters > 0:
                    st.info(
                        f"{active_filters} evaluation filter{'s' if active_filters > 1 else ''} active"
                    )

            with filter_header_col2:
                if active_filters > 0 and st.button("Clear All Filters"):
                    st.session_state.eval_filters = []
                    st.session_state.filtered_samples = {}
                    st.session_state.filters_applied = False
                    st.rerun()

            # Filter mode selection
            filter_mode = st.radio(
                "Filter mode",
                ["ANY", "ALL"],
                horizontal=True,
                format_func=lambda x: (
                    "OR (match any filter)" if x == "ANY" else "AND (match all filters)"
                ),
            )

            filter_col1, filter_col2 = st.columns(2)

            # Filter by number of models that failed
            with filter_col1:
                filter_by_failure_count = st.checkbox("Filter by failure count")
                if filter_by_failure_count:
                    failure_count_operator = st.selectbox(
                        "Operator",
                        ["equal", "greater_than", "less_than"],
                        format_func=lambda x: (
                            "="
                            if x == "equal"
                            else ">=" if x == "greater_than" else "<="
                        ),
                    )
                    failure_count = st.number_input(
                        "Number of models failed",
                        min_value=0,
                        max_value=len(model_names),
                        value=1,
                    )

            # Filter by specific model failure
            with filter_col2:
                filter_by_model_failure = st.checkbox(
                    "Filter by specific model failure"
                )
                failed_model = None
                model_failure_value = None
                if filter_by_model_failure and model_names:
                    failed_model = st.selectbox("Model", model_names)
                    model_failure_value = st.checkbox("Has failed", value=True)

            # Filter by evaluation keys - allow multiple criteria
            st.subheader("Evaluation Key Filters")

            # Display existing filters
            for i, filter_item in enumerate(st.session_state.eval_filters):
                with st.container():
                    cols = st.columns([3, 2, 3, 1])
                    cols[0].text(f"Model: {filter_item['model']}")
                    cols[1].text(f"Key: {filter_item['key']}")
                    cols[2].text(f"Value: {filter_item['value']}")
                    if cols[3].button("❌", key=f"remove_{i}"):
                        st.session_state.eval_filters.pop(i)
                        st.rerun()

            # Add new filter
            with st.expander("Add evaluation filter"):
                eval_cols = st.columns([4, 4, 4])

                new_eval_model = None
                new_eval_key = None
                new_eval_value = None

                if model_names:
                    with eval_cols[0]:
                        new_eval_model = st.selectbox(
                            "Model", model_names, key="new_eval_model"
                        )

                    if new_eval_model and eval_keys:
                        with eval_cols[1]:
                            new_eval_key = st.selectbox(
                                "Evaluation Key", eval_keys, key="new_eval_key"
                            )

                        if new_eval_key:
                            # Get possible values for this evaluation key
                            eval_values = set()
                            for sample_id, sample in samples.items():
                                if new_eval_model in sample.get("model_results", {}):
                                    eval_results = sample["model_results"][
                                        new_eval_model
                                    ].get("evaluation_results", {})
                                    if new_eval_key in eval_results:
                                        if isinstance(eval_results[new_eval_key], str):
                                            eval_values.add(eval_results[new_eval_key])
                                        elif (
                                            isinstance(eval_results[new_eval_key], dict)
                                            and "status" in eval_results[new_eval_key]
                                        ):
                                            eval_values.add(
                                                eval_results[new_eval_key]["status"]
                                            )

                            eval_values = sorted(list(eval_values))
                            if eval_values:
                                with eval_cols[2]:
                                    new_eval_value = st.selectbox(
                                        "Value", eval_values, key="new_eval_value"
                                    )

                # Add filter button
                if (
                    new_eval_model
                    and new_eval_key
                    and new_eval_value
                    and st.button("Add Filter")
                ):
                    st.session_state.eval_filters.append(
                        {
                            "model": new_eval_model,
                            "key": new_eval_key,
                            "value": new_eval_value,
                        }
                    )
                    st.rerun()

            # Apply filters button
            if st.button("Apply Filters"):
                # Prepare filter configuration
                filter_config = {
                    "filter_mode": filter_mode,
                    "filter_by_failure_count": (
                        filter_by_failure_count
                        if "filter_by_failure_count" in locals()
                        else False
                    ),
                    "failure_count_operator": (
                        failure_count_operator
                        if "failure_count_operator" in locals()
                        else "equal"
                    ),
                    "failure_count": (
                        failure_count if "failure_count" in locals() else 1
                    ),
                    "filter_by_model_failure": (
                        filter_by_model_failure
                        if "filter_by_model_failure" in locals()
                        else False
                    ),
                    "failed_model": failed_model,
                    "model_failure_value": (
                        model_failure_value
                        if "model_failure_value" in locals()
                        else True
                    ),
                    "filter_by_evaluation_keys": len(st.session_state.eval_filters) > 0,
                    "eval_filters": st.session_state.eval_filters,
                }

                # Store filter config in session state
                st.session_state.filter_config = filter_config

                # Apply filters
                filtered_samples = filter_samples(benchmark_data, filter_config)

                # Store filtered samples in session state
                st.session_state.filtered_samples = filtered_samples
                st.session_state.filters_applied = True

            # Display filtered samples if filters have been applied
            if st.session_state.filters_applied and st.session_state.filtered_samples:
                st.success(
                    f"Found {len(st.session_state.filtered_samples)} matching samples"
                )

                filtered_sample_ids = list(st.session_state.filtered_samples.keys())
                sample_idx = st.selectbox(
                    "Select a filtered sample",
                    range(len(filtered_sample_ids)),
                    format_func=lambda i: f"Sample {filtered_sample_ids[i]}",
                    key="filtered_sample_selector",
                )

                selected_sample_id = filtered_sample_ids[sample_idx]
                selected_sample = st.session_state.filtered_samples[selected_sample_id]

                # Display the selected sample
                display_sample(selected_sample)
            elif (
                st.session_state.filters_applied
                and not st.session_state.filtered_samples
            ):
                st.warning("No samples match the specified filters")

            # Add a button to reset filters
            if st.session_state.filters_applied:
                if st.button("Clear Results and Reset Filters"):
                    st.session_state.filtered_samples = {}
                    st.session_state.filters_applied = False
                    st.session_state.eval_filters = []
                    st.rerun()


def display_sample(sample):
    """Display a benchmark sample"""
    # Create two columns - system prompt on left, conversation on right
    col1, col2 = st.columns([4, 6])

    with col1:
        st.subheader("System Prompt & Thread")

        # Get the input messages
        input_messages = sample.get("input_messages", [])

        # Look for system message
        system_message = None
        conversation_history = []

        for msg in input_messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                conversation_history.append(msg)

        # Display the system message in an expandable section
        if system_message:
            with st.expander("System Prompt", expanded=False):
                st.text_area("", system_message, height=400)

        # Display the conversation thread
        st.subheader("Conversation Thread")
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                st.markdown("**USER:**")
                st.markdown(f"```\n{content}\n```")
            elif role == "assistant":
                st.markdown("**ASSISTANT:**")
                st.markdown(f"```\n{content}\n```")
            elif role == "tool":
                st.markdown("**TOOL:**")
                st.markdown(f"```\n{content}\n```")

    with col2:
        st.subheader("Model Responses & Evaluation")

        # Display golden response
        golden_response = sample.get("golden_response", "")
        if golden_response:
            with st.expander("Golden Response", expanded=False):
                st.code(golden_response, language="json")

        # Display model results
        model_results = sample.get("model_results", {})

        # Create tabs for each model
        if model_results:
            model_tabs = st.tabs(list(model_results.keys()))

            for i, (model_name, results) in enumerate(model_results.items()):
                with model_tabs[i]:
                    # Check if model failed
                    has_failed = results.get("has_failed", False)

                    if has_failed:
                        st.error("Model failed for this sample")

                    # Display model response
                    model_response = results.get("model_response", "")
                    st.markdown("### Response")
                    st.code(model_response, language="json")

                    # Display evaluation results
                    eval_results = results.get("evaluation_results", {})
                    if eval_results:
                        st.markdown("### Evaluation")

                        # Create expander for each evaluation type
                        for eval_key, eval_data in eval_results.items():
                            with st.expander(f"{eval_key}", expanded=True):
                                if isinstance(eval_data, dict):
                                    # Format as JSON for better readability
                                    st.code(
                                        json.dumps(eval_data, indent=2), language="json"
                                    )
                                else:
                                    st.write(eval_data)


if __name__ == "__main__":
    main()
