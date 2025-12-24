"""
Tests for the config_handler module.
"""

import os
import pytest
import tempfile
import json
from pathlib import Path
from tota_benchmarking.config_handler import (
    load_config,
    validate_models_config,
    get_results_directory,
)


def test_load_config_missing_file():
    """Test that load_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.json")


def test_load_config_invalid_json():
    """Test that load_config raises ValueError for invalid JSON."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_file.write("{ invalid json")
        temp_file_path = temp_file.name

    try:
        with pytest.raises(ValueError):
            load_config(temp_file_path)
    finally:
        os.unlink(temp_file_path)


def test_load_config_missing_required_fields():
    """Test that load_config raises ValueError for missing required fields."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        json.dump({"dataset_name": "test"}, temp_file)
        temp_file_path = temp_file.name

    try:
        with pytest.raises(ValueError):
            load_config(temp_file_path)
    finally:
        os.unlink(temp_file_path)


def test_validate_models_config_empty():
    """Test that validate_models_config raises ValueError for empty config."""
    with pytest.raises(ValueError):
        validate_models_config([])


def test_validate_models_config_missing_fields():
    """Test that validate_models_config raises ValueError for missing fields."""
    with pytest.raises(ValueError):
        validate_models_config([{"name": "Model1"}])


def test_validate_models_config_invalid_provider():
    """Test that validate_models_config raises ValueError for invalid provider."""
    with pytest.raises(ValueError):
        validate_models_config(
            [
                {
                    "name": "Model1",
                    "provider": "INVALID",
                    "base_model_type": "GPT",
                    "path": "/path/to/model",
                }
            ]
        )


def test_validate_models_config_azure_missing_fields():
    """Test that validate_models_config raises ValueError for Azure missing fields."""
    with pytest.raises(ValueError):
        validate_models_config(
            [
                {
                    "name": "Model1",
                    "provider": "AZURE",
                    "base_model_type": "GPT",
                    "path": "/path/to/model",
                }
            ]
        )


def test_get_results_directory():
    """Test that get_results_directory returns a Path and creates the directory."""
    results_dir = get_results_directory()
    assert isinstance(results_dir, Path)
    assert results_dir.exists()
    assert results_dir.is_dir()
