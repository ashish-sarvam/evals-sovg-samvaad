#!/usr/bin/env python
import json
from typing import Any, Dict, List


def convert_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert input/output format to system_prompt/chat_thread/output format."""
    input_messages = data["input"]
    system_prompt = input_messages[0]["content"]
    chat_thread = input_messages[1:]

    output = data["golden_response"]

    return {
        "system_prompt": system_prompt,
        "chat_thread": chat_thread,
        "output": output,
    }


def main(input_file: str, output_file: str):
    converted_data = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            converted = convert_format(data)
            converted_data.append(converted)

    with open(output_file, "w") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert dataset format")
    parser.add_argument("--input-file", help="Path to input JSONL file")
    parser.add_argument("--output-file", help="Path to output JSONL file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    main(input_file, output_file)
