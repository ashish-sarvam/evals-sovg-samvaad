#!/usr/bin/env python3
"""
Script to fix double-encoded JSON in tool_calls arguments.

Converts tool_calls arguments from escaped JSON strings to proper dicts.
Qwen models need arguments as dict, not JSON string.

Usage:
    python fix_tool_content.py -i input.jsonl -o output.jsonl
"""

import argparse
import json
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix double-encoded JSON in tool message content"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--messages-key",
        type=str,
        default="messages",
        help="Key for messages in the JSONL (default: messages)"
    )
    return parser.parse_args()


def decode_until_dict(value: Any, max_depth: int = 5) -> Any:
    """
    Keep decoding JSON strings until we get a dict (or hit max depth).
    
    Args:
        value: The value to decode
        max_depth: Maximum decoding attempts
        
    Returns:
        Decoded value (dict if possible, original otherwise)
    """
    depth = 0
    while isinstance(value, str) and depth < max_depth:
        try:
            decoded = json.loads(value)
            value = decoded
            depth += 1
        except (json.JSONDecodeError, TypeError):
            break
    return value


def fix_tool_calls_args(conversation: dict, messages_key: str = "messages") -> tuple[dict, int]:
    """
    Fix double-encoded JSON in tool_calls arguments.
    
    Args:
        conversation: The conversation dict
        messages_key: Key for messages
        
    Returns:
        Tuple of (fixed conversation, number of fixes)
    """
    messages = conversation.get(messages_key, [])
    fixes = 0
    
    for msg in messages:
        # Fix tool_calls arguments
        if "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                # Handle nested function structure
                if "function" in tool_call and "arguments" in tool_call["function"]:
                    args = tool_call["function"]["arguments"]
                    if isinstance(args, str):
                        decoded = decode_until_dict(args)
                        if isinstance(decoded, dict):
                            tool_call["function"]["arguments"] = decoded
                            fixes += 1
                # Handle flat structure
                elif "arguments" in tool_call:
                    args = tool_call["arguments"]
                    if isinstance(args, str):
                        decoded = decode_until_dict(args)
                        if isinstance(decoded, dict):
                            tool_call["arguments"] = decoded
                            fixes += 1
    
    return conversation, fixes


def process_file(input_path: str, output_path: str, messages_key: str = "messages"):
    """Process the JSONL file and fix tool_calls arguments."""
    
    stats = {
        "total_lines": 0,
        "lines_with_fixes": 0,
        "total_fixes": 0,
        "errors": 0,
    }
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                conversation = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}, skipping: {e}")
                stats["errors"] += 1
                continue
            
            stats["total_lines"] += 1
            
            # Fix tool_calls arguments
            fixed_conv, fixes = fix_tool_calls_args(conversation, messages_key)
            
            if fixes > 0:
                stats["lines_with_fixes"] += 1
                stats["total_fixes"] += fixes
            
            # Write fixed line
            outfile.write(json.dumps(fixed_conv, ensure_ascii=False) + '\n')
            
            # Progress every 10,000 lines
            if stats["total_lines"] % 10000 == 0:
                print(f"Processed {stats['total_lines']} lines... (fixes so far: {stats['total_fixes']})")
    
    return stats


def verify_file(filepath: str, messages_key: str = "messages"):
    """Verify the fixed file - check tool_calls arguments."""
    
    string_args = 0
    dict_args = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                for msg in data.get(messages_key, []):
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            args = None
                            if "function" in tc and "arguments" in tc["function"]:
                                args = tc["function"]["arguments"]
                            elif "arguments" in tc:
                                args = tc["arguments"]
                            
                            if args is not None:
                                if isinstance(args, str):
                                    string_args += 1
                                elif isinstance(args, dict):
                                    dict_args += 1
            except json.JSONDecodeError:
                continue
    
    return string_args, dict_args


def main():
    args = parse_args()
    
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    stats = process_file(args.input, args.output, args.messages_key)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total lines processed: {stats['total_lines']}")
    print(f"  Lines with fixes: {stats['lines_with_fixes']}")
    print(f"  Total tool_calls arguments fixed: {stats['total_fixes']}")
    print(f"  Errors: {stats['errors']}")
    
    # Verify
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    string_count, dict_count = verify_file(args.output, args.messages_key)
    print(f"  Tool args as string: {string_count}")
    print(f"  Tool args as dict: {dict_count}")
    
    if string_count == 0 and dict_count > 0:
        print("  ✓ All tool_calls arguments converted to dict!")
    elif string_count > 0:
        print(f"  ⚠ {string_count} tool_calls still have string arguments")
    
    print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
