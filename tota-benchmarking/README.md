# Tota Benchmarking

A tool for benchmarking models against the Tota (Task-Oriented Tracing and Assessment) dataset.

## Project Structure

The codebase is organized into the following modules:

- `benchmark_runner.py`: Main orchestration logic
- `config_handler.py`: Configuration loading and validation
- `data_processing.py`: Dataset loading and sample processing
- `evaluator.py`: Evaluation of model responses
- `reporting.py`: Results processing and report generation
- `prompt_converters.py`: Adapters for different model prompts and conversation formats
- `models.py`: Client implementations for different model providers
- `schema.py`: Data structures and enums used throughout the codebase

## Setup

### Prerequisites

- Python 3.8 or higher
- Poetry (dependency management)

### Installation Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/tota-benchmarking.git
   cd tota-benchmarking
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Create a configuration file `config.json` in the root directory:

   ```json
   {
     "dataset_name": "org/tota-dataset",
     "revision": "main",
     "models_config": [
       {
         "provider": "SARVAM",
         "url": "https://api.sarvam.ai/v1/chat/completions",
         "path": "sarvamai/tota-v9",
         "name": "Tota v9",
         "base_model_type": "MISTRAL",
         "api_key": "${SARVAM_API_KEY}",
         "structured": false
       }
     ],
     "max_concurrent_tasks": 5,
     "llm_config": {
       "temperature": 0.1,
       "max_tokens": 512
     }
   }
   ```

4. Create a `.env` file in the root directory with your API keys:

   ```
   AZURE_OPENAI_URL=""
   AZURE_OPENAI_API_VERSION=""
   AZURE_OPENAI_MODEL_NAME=""
   AZURE_OPENAI_API_KEY=
   HF_READ_TOKEN=""
   VERTEX_CREDENTIALS="{}"
   ```

5. Activate the Poetry environment:

   ```bash
   poetry shell
   ```

Now you're ready to run the benchmark as described in the "Running the Benchmark" section.

## Configuration

Configuration is managed via the `config.json` file in the root of the project. This file allows you to specify:

- Model configurations
- Dataset parameters
- Concurrency settings
- LLM parameters

### Model Configuration

Each model in the `models_config` array can have the following properties:

- `provider`: The model provider (currently supported: "SARVAM", "AZURE", "OPENAI", "GEMINI", "CLAUDE", "GROQ")
- `url`: The API endpoint URL for the model (for hosted models)
- `path`: The model path or name (to be sent in API args)
- `name`: A reader friendly name for the model
- `base_model_type`: The base model type (e.g., "QWEN", "GEMMA", "MISTRAL", "GPT", "GEMINI", "CLAUDE", "LLAMA")
- `api_key`: (Optional) API key for authentication with bearer token

## Running the Benchmark

To run the benchmark:

```bash
python main.py
```

Command-line options:

```
usage: main.py [-h] [--config CONFIG] [--max-samples MAX_SAMPLES]

TOTA Benchmarking Tool

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file (default: config.json)
  --max-samples MAX_SAMPLES
                        Maximum number of samples to process (default: all)
```

Results will be saved in the `results` directory.

### JSONL Format for Evaluation

The evaluation runner expects a JSONL file with each line containing a JSON object with the following structure:

```json
{
  "id": 123,
  "input": [{ "role": "user", "content": "Question text here" }],
  "golden_response": "The expected correct answer",
  "model_responses": {
    "model_name_1": "Response from first model",
    "model_name_2": "Response from second model"
  }
}
```

Where:

- `id`: Unique identifier for the sample
- `input`: List of conversation messages in the format of role-content pairs
- `golden_response`: The reference answer to evaluate against
- `model_responses`: Dictionary mapping model names to their responses

### Command-line Arguments

When using the evaluation runner through CLI:

```bash
poetry run python main.py --eval-only path/to/samples.jsonl --config config.json --max-samples 100
```

Options:

- `--samples`: Path to the JSONL file with sample data (required)
- `--config`: Path to the configuration file (required)
- `--max-samples`: Maximum number of samples to process (optional)

## Using the Library Programmatically

You can also use the benchmarking library programmatically in your own code:

```python
import asyncio
from tota_benchmarking import run_benchmark

# Run with custom config and only 10 samples
async def main():
    results = await run_benchmark(
        config_path="my_config.json",
        max_samples=10
    )
    print(f"Total samples evaluated: {results['summary']['total_samples']}")

asyncio.run(main())
```

## Development

### Running Tests

```bash
pytest tests/
```

### Adding Support for New Model Providers

To add a new model provider:

1. Update the `models.py` file with the appropriate API client
2. Add the new provider to the validation logic in `config_handler.py`
3. Implement any necessary adapters in `prompt_converters.py`

## Benchmark Dataset and Versions

HF dataset name = `sarvam/tota-evaluation-dataset`

current version = `llama-v0.4`
current version with adequate outputs for some threads = `llama-v0.4-multi`
Number of samples = 496

Next version in progress = `llama-v0.5`
(Adding flows for newly developed and deployed bots - IN PROGRESS)
