"""
Audio Conversion Script for Conversations
Uses Sarvam TTS v3 Beta API (via sarvamai client) to generate audio for
conversation turns and stitches them into a single output file.
"""

import json
import os
from pathlib import Path
from pydub import AudioSegment
import io
import tempfile
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sarvamai import SarvamAI
from sarvamai.play import save
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("SARVAM_API_KEY", "")
MODEL = "bulbul:v3-beta"

# Option 1: Set a folder path to process ALL JSON files in it
CONVERSATION_FOLDER: Optional[str] = str(Path(__file__).parent / "artifacts" / "input")
CONVERSATION_PATHS: List[str] = []  # if not using folder, then use this
OUTPUT_DIR = Path(__file__).parent / "artifacts" / "audio"


USER_SPEAKER = "rahul"  # Female, Natural & Friendly - good for user
ASSISTANT_SPEAKER = "roopa"  # Male, Warm & Energetic - for agent # roopa - female
TARGET_LANGUAGE = "hi-IN"  
TURN_SILENCE_MS = 600

# Parallel generation settings
MAX_WORKERS = 5  # Number of parallel API requests
MAX_RETRIES = 10

# Initialize Sarvam AI client
client = SarvamAI(api_subscription_key=API_KEY)


def get_conversation_paths() -> List[str]:
    """Get list of conversation file paths from folder or explicit list."""
    if CONVERSATION_FOLDER:
        folder = Path(CONVERSATION_FOLDER)
        if folder.exists() and folder.is_dir():
            # Get all JSON files in the folder
            paths = sorted([str(f) for f in folder.glob("*.json")])
            print(f"Found {len(paths)} JSON files in {CONVERSATION_FOLDER}")
            return paths
        else:
            print(f"Warning: Folder not found: {CONVERSATION_FOLDER}")
            return []
    return CONVERSATION_PATHS


def load_conversation(file_path: str) -> Dict:
    """Load a conversation JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS by removing newlines, tabs, and extra whitespace.
    """
    # Remove newlines and tabs
    cleaned = text.replace("\n", " ").replace("\t", " ")
    # Collapse multiple spaces into single space
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def extract_speakable_turns(conversation: Dict) -> List[Dict]:
    """
    Extract turns that should be converted to speech.
    Skips tool turns and assistant turns with empty content.
    """
    # Support both "turns" and "messages" keys
    turns = conversation.get("turns") or conversation.get("messages", [])
    speakable_turns = []

    for turn in turns:
        role = turn.get("role")
        content = turn.get("content", "").strip()

        # Skip tool turns
        if role == "tool":
            continue

        # Skip turns with empty content (assistant turns with only tool_calls)
        if not content:
            continue

        # Only include user and assistant turns with content
        if role in ["user", "assistant"]:
            # Clean the content for TTS (remove \n, \t, extra spaces)
            cleaned_content = clean_text_for_tts(content)
            speakable_turns.append({"role": role, "content": cleaned_content})

    return speakable_turns


def generate_audio_for_text(
    text: str, speaker: str
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Generate audio for a given text using Sarvam TTS client.
    Returns tuple of (audio bytes, error message). On success, error is None.
    """
    try:
        audio = client.text_to_speech.convert(
            target_language_code=TARGET_LANGUAGE,
            text=text,
            model=MODEL,
            speaker=speaker,
        )

        # Save to a temp file and read back as bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        save(audio, tmp_path)

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        return audio_bytes, None

    except Exception as e:
        error_str = str(e)
        # Try to extract the error message from the body if present
        error_reason = error_str
        if "body:" in error_str:
            try:
                # Extract the body part and parse the message
                body_part = error_str.split("body:")[1].strip()
                import ast

                body_dict = ast.literal_eval(body_part)
                if "error" in body_dict and "message" in body_dict["error"]:
                    error_reason = body_dict["error"]["message"]
            except Exception:
                pass  # Use full error string if parsing fails
        print(f"API Error: {e}")
        return None, error_reason


def generate_audio_for_turn(
    turn: Dict, turn_index: int
) -> tuple[Optional[Dict], Optional[str]]:
    """
    Generate audio for a single conversation turn.
    Returns tuple of (dict with turn info and audio bytes, error message).
    On success, error is None.
    """
    role = turn["role"]
    content = turn["content"]

    # Select speaker based on role
    speaker = USER_SPEAKER if role == "user" else ASSISTANT_SPEAKER

    preview = content[:50]
    print(f"  [{turn_index + 1}] {role} ({speaker}): {preview}...")

    audio_bytes, error_reason = generate_audio_for_text(content, speaker)

    if audio_bytes:
        return {
            "index": turn_index,
            "role": role,
            "speaker": speaker,
            "content": content,
            "audio": audio_bytes,
        }, None
    return None, error_reason


def stitch_audio_segments(audio_results: List[Dict], output_path: str) -> bool:
    """
    Stitch multiple audio segments into a single output file.
    Adds silence between turns for natural conversation flow.
    """
    try:
        # Sort by index to maintain conversation order
        audio_results.sort(key=lambda x: x["index"])

        # Debug: print the order
        print("  Stitching order:")
        for result in audio_results:
            preview = result["content"][:40].replace("\n", " ")
            print(f"    [{result['index']}] {result['role']}: {preview}...")

        # Create silence segment
        silence = AudioSegment.silent(duration=TURN_SILENCE_MS)

        # Initialize combined audio
        combined = AudioSegment.empty()

        for i, result in enumerate(audio_results):
            # Try WAV first, then MP3 as fallback (sarvamai may return either)
            audio_bytes = io.BytesIO(result["audio"])
            try:
                audio_segment = AudioSegment.from_wav(audio_bytes)
            except Exception:
                audio_bytes.seek(0)
                try:
                    audio_segment = AudioSegment.from_mp3(audio_bytes)
                except Exception:
                    audio_bytes.seek(0)
                    # Final fallback: let pydub auto-detect
                    audio_segment = AudioSegment.from_file(audio_bytes)

            # Add silence before (except for first turn)
            if i > 0:
                combined += silence

            # Add the audio segment
            combined += audio_segment

        # Export combined audio
        combined.export(output_path, format="mp3")
        print(f"✓ Stitched audio saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error stitching audio: {e}")
        import traceback

        traceback.print_exc()
        return False


def process_conversation(
    file_path: str, output_dir: Path = OUTPUT_DIR
) -> Optional[str]:
    """
    Process a single conversation file:
    1. Load and extract speakable turns
    2. Generate audio for each turn
    3. Stitch into a single output file

    Returns the output file path on success.
    """
    print("\n" + "=" * 60)
    print(f"Processing: {file_path}")
    print("=" * 60)

    # Load conversation
    try:
        conversation = load_conversation(file_path)
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return None

    # Extract speakable turns
    turns = extract_speakable_turns(conversation)
    print(f"Found {len(turns)} speakable turns")

    if not turns:
        print("No speakable turns found, skipping...")
        return None

    # Generate audio for each turn in parallel
    print(f"\nGenerating audio in parallel ({MAX_WORKERS} workers)...")

    # Store results by index to ensure all are present
    audio_results_dict: Dict[int, Dict] = {}
    failed_indices: List[int] = []
    failure_reasons: Dict[int, str] = {}  # Track failure reasons per turn

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(generate_audio_for_turn, turn, i): i
            for i, turn in enumerate(turns)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result, error_reason = future.result()
                if result:
                    audio_results_dict[idx] = result
                else:
                    reason_msg = f" ({error_reason})" if error_reason else ""
                    print(f"  ✗ Failed turn {idx + 1}{reason_msg}")
                    failed_indices.append(idx)
                    if error_reason:
                        failure_reasons[idx] = error_reason
            except Exception as e:
                print(f"  ✗ Error generating turn {idx + 1}: {e}")
                failed_indices.append(idx)
                failure_reasons[idx] = str(e)

    # Retry failed turns sequentially
    for retry in range(MAX_RETRIES):
        if not failed_indices:
            break
        num_failed = len(failed_indices)
        print(
            f"\n  Retrying {num_failed} failed turns (attempt {retry + 1})..."
        )
        still_failed = []
        for idx in failed_indices:
            result, error_reason = generate_audio_for_turn(turns[idx], idx)
            if result:
                audio_results_dict[idx] = result
                print(f"    ✓ Retry succeeded for turn {idx + 1}")
                failure_reasons.pop(idx, None)  # Clear on success
            else:
                still_failed.append(idx)
                reason_msg = f" ({error_reason})" if error_reason else ""
                print(f"    ✗ Retry failed turn {idx + 1}{reason_msg}")
                if error_reason:
                    failure_reasons[idx] = error_reason
        failed_indices = still_failed

    # Check if all turns were generated
    total_count = len(turns)
    success_count = len(audio_results_dict)
    print(f"\nGenerated audio for {success_count}/{total_count} turns")

    if success_count < total_count:
        missing = [
            i + 1 for i in range(total_count) if i not in audio_results_dict
        ]
        print(f"  ⚠ MISSING TURNS: {missing}")
        print("  Aborting - not all turns were generated!")
        return None

    # Validate all indices are present (0 to total_count-1)
    expected_indices = set(range(total_count))
    actual_indices = set(audio_results_dict.keys())
    if expected_indices != actual_indices:
        print(f"  ⚠ Index mismatch! Expected {expected_indices}")
        print(f"    Got: {actual_indices}")
        return None

    # Convert dict to sorted list
    audio_results = [audio_results_dict[i] for i in range(total_count)]
    print(f"  ✓ All {total_count} turns ready for stitching")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename from input filename
    input_name = Path(file_path).stem
    output_path = output_dir / f"{input_name}_audio.mp3"

    # Stitch audio segments
    print("\nStitching audio segments...")
    if stitch_audio_segments(audio_results, str(output_path)):
        return str(output_path)

    return None


def process_all_conversations(
    paths: Optional[List[str]] = None, output_dir: Path = OUTPUT_DIR
) -> Dict[str, str]:
    """
    Process all conversation files in the paths list.
    If paths is None, uses get_conversation_paths() to find files.
    Returns a dict mapping input paths to output paths.
    """
    if paths is None:
        paths = get_conversation_paths()

    results = {}

    print("\n" + "#" * 60)
    print(f"Processing {len(paths)} conversation(s)")
    print(f"User Speaker: {USER_SPEAKER}")
    print(f"Assistant Speaker: {ASSISTANT_SPEAKER}")
    print(f"Output Directory: {output_dir}")
    print("#" * 60)

    for path in paths:
        output_path = process_conversation(path, output_dir)
        if output_path:
            results[path] = output_path

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {len(results)}/{len(paths)} conversations")

    if results:
        print("\nOutput files:")
        for input_path, output_path in results.items():
            print(f"  {Path(input_path).name} -> {output_path}")

    return results


# ============================================================================
# TEST FUNCTION
# ============================================================================


def test_api():
    """
    Test the Sarvam TTS API with a simple request using the client.
    """
    print("=" * 60)
    print("TESTING SARVAM TTS API (Client)")
    print("=" * 60)

    test_text = "Hello, this is a test message."

    print(f"\nTest text: {test_text}")
    print(f"Speaker: {ASSISTANT_SPEAKER}")
    print(f"Language: {TARGET_LANGUAGE}")

    try:
        print("\nGenerating audio...")
        audio_bytes, error_reason = generate_audio_for_text(
            test_text, ASSISTANT_SPEAKER
        )

        if audio_bytes:
            print(f"\n✓ SUCCESS! Received {len(audio_bytes)} bytes of audio")

            # Save test audio
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            test_output = OUTPUT_DIR / "test_audio.wav"
            with open(test_output, "wb") as f:
                f.write(audio_bytes)
            print(f"✓ Test audio saved to: {test_output}")
            return True
        else:
            reason_msg = f" (Reason: {error_reason})" if error_reason else ""
            print(f"\n✗ Failed to generate audio{reason_msg}")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check for --test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_api()
    else:
        paths = get_conversation_paths()
        if not paths:
            print("No conversation files found!")
            print("\nSet CONVERSATION_FOLDER to a directory with JSON files,")
            print("or add paths to CONVERSATION_PATHS list.")
            print("\nExample:")
            print('CONVERSATION_FOLDER = "/path/to/conversations"')
            print("# or")
            print("CONVERSATION_PATHS = [")
            print('    "/path/to/conversation1.json",')
            print('    "/path/to/conversation2.json",')
            print("]")
        else:
            process_all_conversations(paths)
