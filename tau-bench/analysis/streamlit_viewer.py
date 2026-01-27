#!/usr/bin/env python3
# Copyright Sierra
"""
Streamlit viewer for tau-bench comparison results.

Usage:
    streamlit run analysis/streamlit_viewer.py -- --comparison results/comparisons/comparison.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import streamlit as st
except ImportError:
    print("Please install streamlit: pip install streamlit")
    sys.exit(1)

# Status colors and labels
STATUS_CONFIG = {
    "both_pass": {"color": "#22c55e", "label": "✅ Both Pass", "bg": "#dcfce7"},
    "both_fail": {"color": "#ef4444", "label": "❌ Both Fail", "bg": "#fee2e2"},
    "model2_only_fail": {"color": "#3b82f6", "label": "🔵 Model2 Only Fail", "bg": "#dbeafe"},
    "model1_only_fail": {"color": "#f97316", "label": "🟠 Model1 Only Fail", "bg": "#ffedd5"},
}

FAILURE_COLORS = {
    "success": "#22c55e",
    "called_wrong_tool": "#ef4444",
    "used_wrong_tool_argument": "#f97316",
    "goal_partially_completed": "#eab308",
    "authentication_failure": "#8b5cf6",
    "premature_termination": "#ec4899",
    "excessive_tool_calls": "#06b6d4",
    "missing_tool_calls": "#f43f5e",
    "wrong_arguments": "#fb923c",
    "no_trajectory": "#6b7280",
    "other": "#6b7280",
}


def load_comparison_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load comparison results from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def format_tool_call(tc: Dict[str, Any]) -> str:
    """Format a tool call for display."""
    func = tc.get("function", {})
    name = func.get("name", "unknown")
    args = func.get("arguments", "")
    
    try:
        if isinstance(args, str):
            args_dict = json.loads(args)
            args_str = json.dumps(args_dict, indent=2)
        else:
            args_str = json.dumps(args, indent=2)
    except:
        args_str = str(args)
    
    return f"**🔧 {name}**\n```json\n{args_str}\n```"


def render_message(msg: Dict[str, Any], is_failing: bool = False):
    """Render a single message."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])
    turn = msg.get("turn", "?")
    
    role_icons = {"user": "👤", "assistant": "🤖", "tool": "🔧"}
    icon = role_icons.get(role, "❓")
    
    # Container styling
    if is_failing:
        st.markdown(
            f"""<div style="background: #fee2e2; border-left: 4px solid #ef4444; 
            padding: 10px; margin: 5px 0; border-radius: 4px;">
            <strong>⚠️ FAILING TURN {turn}: {icon} {role.upper()}</strong>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"**Turn {turn}: {icon} {role.upper()}**")
    
    if content:
        # Truncate very long content
        if len(content) > 3000:
            content = content[:3000] + "\n\n... (truncated)"
        st.markdown(content)
    
    if tool_calls:
        for tc in tool_calls:
            st.markdown(format_tool_call(tc))
    
    st.divider()


def render_trajectory(traj: List[Dict[str, Any]], label: str, is_success: bool):
    """Render a full trajectory."""
    status_icon = "✅" if is_success else "❌"
    st.subheader(f"{status_icon} {label}")
    
    if not traj:
        st.info("No trajectory available")
        return
    
    for msg in traj:
        is_failing = msg.get("is_failing_turn", False)
        render_message(msg, is_failing)


def render_failure_badge(ftype: str):
    """Render a failure type badge."""
    color = FAILURE_COLORS.get(ftype, "#6b7280")
    label = ftype.replace("_", " ").title()
    st.markdown(
        f'<span style="background: {color}; color: white; padding: 4px 12px; '
        f'border-radius: 4px; font-size: 14px;">{label}</span>',
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title="τ-Bench Comparison Viewer",
        page_icon="🔬",
        layout="wide",
    )
    
    st.title("🔬 τ-Bench Comparison Viewer")
    st.markdown("Compare model trajectories and analyze failures side-by-side.")
    
    # File upload or path from args
    comparison_file = None
    
    # Check command line args
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--comparison", "-c", type=str)
        args, _ = parser.parse_known_args()
        if args.comparison:
            comparison_file = args.comparison
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Comparison JSON", type=["json"])
    
    if uploaded_file:
        data = json.load(uploaded_file)
    elif comparison_file:
        data = load_comparison_file(comparison_file)
        st.success(f"Loaded: {comparison_file}")
    else:
        st.info("Please upload a comparison JSON file or provide one via --comparison argument")
        st.stop()
    
    if not data:
        st.stop()
    
    # Summary Section
    st.header("📊 Summary")
    
    summary = data.get("summary", {})
    model1_name = data.get("model1_name", "Model 1")
    model2_name = data.get("model2_name", "Model 2")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model 1", model1_name)
        st.metric("Pass Rate", f"{summary.get('model1_pass_rate', 0) * 100:.1f}%")
    with col2:
        st.metric("Model 2", model2_name)
        st.metric("Pass Rate", f"{summary.get('model2_pass_rate', 0) * 100:.1f}%")
    
    # Status breakdown
    st.subheader("Status Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Both Pass", summary.get("both_pass", 0))
    with col2:
        st.metric("❌ Both Fail", summary.get("both_fail", 0))
    with col3:
        st.metric("🔵 M2 Only Fail", summary.get("model2_only_fail", 0))
    with col4:
        st.metric("🟠 M1 Only Fail", summary.get("model1_only_fail", 0))
    
    st.markdown(f"**Total tasks compared:** {summary.get('total_tasks_compared', 0)}")
    
    # Task explorer
    st.header("🔍 Task Explorer")
    
    comparisons = data.get("comparisons", [])
    
    # Filters
    col1, col2 = st.columns([1, 3])
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["all", "both_pass", "both_fail", "model2_only_fail", "model1_only_fail"],
            format_func=lambda x: {
                "all": "All Tasks",
                "both_pass": "✅ Both Pass",
                "both_fail": "❌ Both Fail",
                "model2_only_fail": "🔵 Model2 Only Fail",
                "model1_only_fail": "🟠 Model1 Only Fail",
            }.get(x, x)
        )
    
    # Filter comparisons
    filtered = [
        c for c in comparisons 
        if status_filter == "all" or c.get("status") == status_filter
    ]
    
    with col2:
        task_options = {
            f"Task {c['task_id']} - {STATUS_CONFIG.get(c['status'], {}).get('label', c['status'])}": c['task_id']
            for c in filtered
        }
        if task_options:
            selected_task_label = st.selectbox("Select Task", options=list(task_options.keys()))
            selected_task_id = task_options[selected_task_label]
        else:
            st.warning("No tasks match the filter")
            st.stop()
    
    # Get selected task
    task = next((c for c in comparisons if c.get("task_id") == selected_task_id), None)
    if not task:
        st.error("Task not found")
        st.stop()
    
    # Task info
    st.subheader("📋 Task Information")
    
    status = task.get("status", "")
    status_cfg = STATUS_CONFIG.get(status, {"label": status, "bg": "#gray"})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Task ID", task.get("task_id"))
    with col2:
        st.markdown(f"**Status:** {status_cfg['label']}")
    with col3:
        st.metric("Model 1 Reward", f"{task.get('model1_reward', 0):.2f}")
    with col4:
        st.metric("Model 2 Reward", f"{task.get('model2_reward', 0):.2f}")
    
    # Instruction
    with st.expander("📝 Task Instruction", expanded=True):
        st.markdown(task.get("instruction", "No instruction available"))
    
    # Ground truth
    actions = task.get("ground_truth_actions", [])
    if actions:
        with st.expander("🎯 Ground Truth Actions"):
            for a in actions:
                st.markdown(f"- {a}")
    
    # Failure analysis side by side
    st.subheader("🔬 Failure Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{model1_name}**")
        m1_type = task.get("model1_failure_type", "unknown")
        render_failure_badge(m1_type)
        st.markdown(f"*{task.get('model1_failure_description', '')}*")
    
    with col2:
        st.markdown(f"**{model2_name}**")
        m2_type = task.get("model2_failure_type", "unknown")
        render_failure_badge(m2_type)
        st.markdown(f"*{task.get('model2_failure_description', '')}*")
    
    # Trajectories side by side
    st.subheader("📊 Trajectory Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        m1_traj = task.get("model1_trajectory", [])
        render_trajectory(m1_traj, model1_name, m1_type == "success")
    
    with col2:
        m2_traj = task.get("model2_trajectory", [])
        render_trajectory(m2_traj, model2_name, m2_type == "success")
    
    # Failure type legend
    with st.expander("📖 Failure Type Legend"):
        legend_data = {
            "Failure Type": list(FAILURE_COLORS.keys()),
            "Description": [
                "Task completed successfully",
                "Agent called a tool that wasn't appropriate",
                "Agent used incorrect arguments when calling a tool",
                "Agent completed some but not all required actions",
                "Agent failed to verify user identity before sensitive ops",
                "Agent ended conversation before completing task",
                "Agent made too many tool calls (>50)",
                "Agent didn't call required tools",
                "Agent used wrong argument values",
                "No trajectory was recorded",
                "Failure doesn't fit other categories",
            ]
        }
        st.table(legend_data)


if __name__ == "__main__":
    main()

