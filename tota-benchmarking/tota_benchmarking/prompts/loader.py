import os
import pathlib
from typing import Optional
import jinja2


def load_template(
    template_name: str, templates_dir: Optional[str] = None
) -> jinja2.Template:
    """
    Load a template file from the templates directory.

    Args:
        template_name: Name of the template file to load (with or without extension)
        templates_dir: Optional custom templates directory path. If not provided,
                      will use the default 'templates' directory relative to this file.

    Returns:
        The content of the template file as a string

    Raises:
        FileNotFoundError: If the template file cannot be found
    """
    # Add .j2 extension if not already present
    if not template_name.endswith(".j2"):
        template_name = f"{template_name}.j2"

    # Determine the templates directory path
    if templates_dir is None:
        # Get the directory of the current file
        current_dir = pathlib.Path(__file__).parent.absolute()
        # Templates directory is a subdirectory named 'templates'
        templates_dir = os.path.join(current_dir, "templates")

    # Construct the full path to the template file
    template_path = os.path.join(templates_dir, template_name)

    # Check if the file exists
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Read and return the template content
    with open(template_path, "r", encoding="utf-8") as file:
        return jinja2.Template(file.read())
