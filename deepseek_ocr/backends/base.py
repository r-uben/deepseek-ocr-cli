"""Abstract base class for OCR backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from PIL import Image


class Backend(ABC):
    """Abstract base class defining the backend interface for OCR processing."""

    # Default prompts for different OCR tasks
    PROMPTS = {
        "convert": "<|grounding|>Convert the document to markdown.",
        "ocr": "Free OCR.",
        "layout": "<|grounding|>Given the layout of the image.",
        "extract": "Extract the text in the image.",
        "parse": "Parse the figure.",
        "describe_figure": (
            "Describe this figure in detail. What does the chart/graph/diagram show? "
            "Explain the axes, data, and key findings."
        ),
    }

    def __init__(
        self,
        model_name: str,
        max_dimension: int | None = None,
    ):
        self.model_name = model_name
        self.max_dimension = max_dimension
        self.model: bool = False

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend identifier (e.g., 'ollama', 'vllm')."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Initialize connection to the backend and verify model availability."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Clean up backend connection."""
        ...

    @abstractmethod
    def process_image(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str | None = None,
        task: str = "convert",
        return_raw: bool = False,
    ) -> str:
        """Process a single image and return OCR text.

        Args:
            image: PIL Image, file path, or path string
            prompt: Custom prompt (overrides task prompt)
            task: Task type (convert, ocr, layout, extract, parse)
            return_raw: If True, return raw model output without cleaning

        Returns:
            OCR text result
        """
        ...

    def process_images_batch(
        self,
        images: list,
        prompt: str | None = None,
        task: str = "convert",
    ) -> list:
        """Process multiple images sequentially.

        Default implementation; backends can override for optimization.
        """
        results = []
        for image in images:
            result = self.process_image(image, prompt=prompt, task=task)
            results.append(result)
        return results

    def describe_figure(self, image: Union[Image.Image, Path, str]) -> str:
        """Generate a description of a figure/chart/diagram."""
        return self.process_image(
            image,
            prompt=self.PROMPTS["describe_figure"],
            task="describe_figure",
        )

    def get_prompt(self, task: str) -> str:
        """Get the default prompt for a task."""
        return self.PROMPTS.get(task, self.PROMPTS["convert"])
