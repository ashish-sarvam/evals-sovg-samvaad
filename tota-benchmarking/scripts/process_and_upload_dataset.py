#!/usr/bin/env python3
"""
Script to clean a JSONL dataset and upload it to Hugging Face datasets.
The script performs the following operations:
1. Cleans the dataset (default behavior)
2. Uploads the cleaned dataset to Hugging Face Hub
"""

import json
import argparse
from typing import Optional, Dict, Any
from collections import defaultdict
from datasets import load_dataset
import re
import ast


def extract_template_variables(
    template_content, example_type=None, expected_output=None
):
    """Extract variables from a Jinja2 template."""
    variables = {}

    # Add example type and expected output if provided
    if example_type:
        variables["example_type"] = example_type

    # Extract current_language
    current_language_match = re.search(
        r"inform the user that you are speaking in ([^,]+),", template_content
    )
    if current_language_match:
        variables["current_language"] = current_language_match.group(1).strip()
    else:
        variables["current_language"] = ""

    # Check if enable_agentic_lid is true
    enable_agentic_lid = "GI.7" in template_content or "GI.8" in template_content
    variables["enable_agentic_lid"] = enable_agentic_lid
    # Extract languages_available
    languages_available_match = re.search(
        r"inform the user that you can speak (.*?)(?:, and|\.)", template_content
    )
    if languages_available_match:
        languages_str = languages_available_match.group(1).strip()
        # Convert to list of strings
        variables["languages_available"] = [
            lang.strip() for lang in languages_str.split(",")
        ]
    else:
        variables["languages_available"] = []

    # Extract global_prompt - Found in "# General bot instructions" section
    pattern = r"# General bot instructions\s+(.*?)(?=\s*#\s+[A-Za-z]|\Z)"

    # Use re.DOTALL to make . match newline characters as well
    global_prompt_match = re.search(pattern, template_content, re.DOTALL)
    if global_prompt_match:
        variables["global_prompt"] = (
            global_prompt_match.group(1)
            .strip()
            .replace("\\n", "\n")
            .replace('\\"', '"')
        )
    else:
        variables["global_prompt"] = ""

    # Extract instructions - Found in "# State specific instructions" section
    instructions_pattern = r"# State specific instructions([\s\S]*?)(?=\n# [^#]|$)"
    instructions_match = re.search(instructions_pattern, template_content, re.DOTALL)

    if instructions_match:
        content = instructions_match.group(1).strip()
        content = content.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        variables["instructions"] = content
    else:
        variables["instructions"] = ""

    kb_details_pattern = r"# KB details for retrieval:\s*([\s\S]*?)\s*Note that if the query is related to the above description"  # noqa

    # Extract kb_details (if present)
    kb_details_match = re.search(kb_details_pattern, template_content, re.DOTALL)
    if kb_details_match:
        content = kb_details_match.group(1).strip()
        content = content.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        variables["kb_details"] = content
    else:
        variables["kb_details"] = ""

    immutable_pattern = r"## Constants non-editable\s*([\s\S]*?)(?=## Editable variables with thier current value|$)"  # noqa
    # Extract immutable_variables
    immutable_vars_match = re.search(
        immutable_pattern,
        template_content,
        re.DOTALL,
    )
    if immutable_vars_match:
        content = immutable_vars_match.group(1).strip()
        content = content.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        # Convert to list of dictionaries
        immutable_variables = []
        for line in content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                immutable_variables.append((key.strip(), value.strip()))
        variables["immutable_variables"] = immutable_variables
    else:
        variables["immutable_variables"] = []

    # Extract mutable_variables_with_values
    mutable_pattern = r"## Editable variables with their current value\s*([\s\S]*?)(?=\n(?:Current State:|Available state transitions:|## |$))"  # noqa
    mutable_vars_match = re.search(
        mutable_pattern,
        template_content,
        re.DOTALL,
    )
    if mutable_vars_match:
        content = mutable_vars_match.group(1).strip()
        content = content.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        # Convert to list of dictionaries
        mutable_variables = []
        for line in content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                mutable_variables.append((key.strip(), value.strip()))
        variables["mutable_variables_with_values"] = mutable_variables
    else:
        variables["mutable_variables_with_values"] = []
    # Extract name (current state)
    name_match = re.search(r"Current State: (.*?) \(You are already", template_content)
    if name_match:
        variables["current_state_name"] = name_match.group(1).strip()
    else:
        variables["current_state_name"] = ""
    # Extract next_states
    next_states_match = re.search(
        r"Available state transitions: (.*?) \(Transition to", template_content
    )
    if next_states_match:
        states_str = next_states_match.group(1).strip()
        try:
            variables["next_states"] = list(ast.literal_eval(states_str))
        except Exception:
            variables["next_states"] = [t.strip() for t in states_str.split(",")]

    elif "This is a terminal state. No state transition possible." in template_content:
        variables["next_states"] = []

    tools_pattern = r"# Available tools\s*(.*?)(?=\n\n|$)"
    # Extract tools
    tools_match = re.search(tools_pattern, template_content, re.DOTALL)
    if tools_match:
        tools_str = tools_match.group(1).strip()
        if tools_str.startswith("[") and tools_str.endswith("]"):
            # It's a list in square brackets
            try:
                variables["tools"] = ast.literal_eval(tools_str)
            except Exception:
                # If parsing fails, fall back to string splitting
                tools_str = tools_str.strip("[]")
                variables["tools"] = [t.strip() for t in tools_str.split(",")]
        else:
            # It's a comma-separated string
            variables["tools"] = [t.strip() for t in tools_str.split(",")]
    else:
        variables["tools"] = []

    if len(variables["tools"]) > 0:
        variables["needs_tool_calls_nl"] = True
    else:
        variables["needs_tool_calls_nl"] = False

    if len(variables["next_states"]) > 0:
        variables["state_transition"] = True
    else:
        variables["state_transition"] = False

    if (
        len(variables["immutable_variables"]) > 0
        or len(variables["mutable_variables_with_values"]) > 0
    ):
        variables["variables"] = True
    else:
        variables["variables"] = False

    return variables


def clean_dataset(input_path: str, output_path: str) -> None:
    """
    Clean a JSONL dataset by normalizing data types and extracting template variables.

    Args:
        input_path: Path to the input JSONL file
        output_path: Path to save the cleaned JSONL file
    """
    print(f"Cleaning dataset from {input_path}...")
    cleaned_data = []

    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if "output" in data:
                if isinstance(data["output"], dict):
                    data["output"] = (
                        json.dumps(data["output"])
                        .replace('": ', '":')
                        .replace(', "', ',"')
                    )

            if "input_messages" in data:
                input_thread = data["input_messages"]
                system_message = input_thread[0]["content"]
                chat_thread = input_thread[1:]

                template_vars = extract_template_variables(system_message)

                data.pop("input_messages")
                data["chat_thread"] = chat_thread
                data = data | template_vars

            cleaned_data.append(data)

    type_consistency = defaultdict(set)
    type_examples = defaultdict(list)

    for row_idx, row in enumerate(cleaned_data):
        for key, value in row.items():
            type_consistency[key].add(type(value).__name__)
            type_examples[key].append((row_idx, value, type(value).__name__))

    print("\nData Type Consistency Report:")
    print("-" * 50)
    for key, types in type_consistency.items():
        if len(types) > 1:
            print(f"\n⚠️  {key}: Inconsistent types found - {types}")
            print("Examples of inconsistent data points:")
            type_groups = defaultdict(list)
            for row_idx, value, value_type in type_examples[key]:
                type_groups[value_type].append((row_idx, value))

            for value_type, examples in type_groups.items():
                print(f"\nType: {value_type}")
                # Show up to 3 examples for each type
                for row_idx, value in examples[:3]:
                    print(f"  Row {row_idx}: {value}")
        else:
            print(f"✅ {key}: Consistent type - {types.pop()}")

    print("\nOriginal keys in first row:")
    print(cleaned_data[0].keys())

    for row in cleaned_data:
        instructions = row["instructions"]
        output = row["output"]
        output_dict = json.loads(output)
        if "lines" not in output_dict:
            continue

        si_instructions = []
        for line in output_dict["lines"]:
            if line.strip().startswith("SI"):
                si_instructions.append(line)

        if len(si_instructions) == 0:
            continue

        # Extract the x value from each SI instruction
        si_numbers = []
        for si in si_instructions:
            # Extract the SI number (x) from SIx.y1.y2.y3...
            si_match = re.search(r'SI(\d+)', si)
            if si_match:
                si_number = int(si_match.group(1))
                si_numbers.append(si_number)

        # Extract SI number from instructions
        instruction_si_number = None
        if instructions:
            # Split instructions by newline
            instruction_lines = instructions.split('\n')
            # Find the first line that starts with SI
            for line in instruction_lines:
                if line.strip().startswith('SI'):
                    # Extract the SI number (x) from SIx.y1.y2.y3...
                    si_match = re.search(r'SI(\d+)', line)
                    if si_match:
                        instruction_si_number = int(si_match.group(1))
                        break

        # Sanity check: Compare SI numbers from instructions and output
        if instruction_si_number is not None and si_numbers:
            if instruction_si_number not in si_numbers:
                print(f"⚠️  Mismatch found: Instructions SI{instruction_si_number} not found in output SI numbers {si_numbers}")
            else:
                # Optional: Uncomment if you want to see matches
                print(f"✅ Match found: Instructions SI{instruction_si_number} found in output SI numbers {si_numbers}")

    # Write the cleaned data to a new JSONL file
    with open(output_path, "w") as f:
        for item in cleaned_data:
            f.write(json.dumps(item) + "\n")

    print(f"Cleaned dataset saved to {output_path}")


def upload_dataset(
    jsonl_path: str,
    dataset_name: str,
    token: str,
    revision: Optional[str] = None,
    private: bool = False,
    description: Optional[str] = None,
) -> None:
    """
    Upload a JSONL dataset to Hugging Face Hub.

    Args:
        jsonl_path: Path to the JSONL file
        dataset_name: Name of the dataset on the Hub (username/dataset_name)
        token: Hugging Face API token
        revision: Optional branch or tag name
        private: Whether the dataset should be private
        description: Optional description for the dataset
    """
    # Load data directly from JSONL file
    print(f"Loading dataset from {jsonl_path}...")
    dataset = load_dataset("json", data_files=jsonl_path)
    print(f"Loaded dataset with {len(dataset['train'])} examples.")

    # Prepare upload arguments
    upload_kwargs: Dict[str, Any] = {
        "private": private,
    }

    if revision:
        upload_kwargs["revision"] = revision

    if description:
        upload_kwargs["description"] = description

    # Upload to Hub
    print(f"Uploading dataset to {dataset_name}...")
    dataset.push_to_hub(dataset_name, token=token, **upload_kwargs)

    print(f"Dataset uploaded successfully to {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean a JSONL dataset and upload it to Hugging Face Hub"
    )
    parser.add_argument(
        "--input-path", required=True, help="Path to the input JSONL file"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save the cleaned JSONL file (default: ./cleaned_dataset.jsonl)",
        default="./cleaned_dataset.jsonl",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of the dataset on the Hub (username/dataset_name)",
    )
    parser.add_argument("--token", help="Hugging Face API token")
    parser.add_argument("--revision", help="Branch or tag name (optional)")
    parser.add_argument(
        "--private", action="store_true", help="Whether the dataset should be private"
    )
    parser.add_argument("--description", help="Description for the dataset")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip cleaning and only upload the dataset",
    )

    args = parser.parse_args()

    # Clean the dataset (default behavior)
    if not args.upload_only:
        clean_dataset(args.input_path, args.output_path)
    else:
        print("Skipping cleaning step as --upload-only flag is set")

    # Upload the dataset if dataset_name and token are provided
    if args.dataset_name and args.token:
        upload_dataset(
            jsonl_path=args.output_path,
            dataset_name=args.dataset_name,
            token=args.token,
            revision=args.revision,
            private=args.private,
            description=args.description,
        )
    else:
        print("Skipping upload step as --dataset-name and --token are not provided")


if __name__ == "__main__":
    main()
