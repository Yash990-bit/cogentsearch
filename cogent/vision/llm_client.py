"""
Vision LLM Client Module for Multimodal Visual Reasoning.
Provides unified interfaces to OpenAI (GPT-4o / GPT-4o-mini) and Anthropic (Claude 3.5 Sonnet)
with automatic offline/mock fallback for developer onboarding and automated CI/CD testing.
"""

from __future__ import annotations
import base64
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from loguru import logger

from cogent.core.config import settings
from cogent.schemas.document import VisualElementType


SYSTEM_VISION_PROMPT = """You are an Expert Multimodal Document Intelligence Analyst.
Your task is to analyze the provided cropped visual asset from a document (which may be a table, chart, architectural diagram, flowchart, photo, or infographic).

Provide a comprehensive, high-fidelity response formatted strictly as follows:
1. ELEMENT_TYPE: Choose one of [table, chart, diagram, infographic, image]
2. SUMMARY: A concise 2-3 sentence overview describing what this visual element represents and its purpose in the document.
3. KEY_DATA_POINTS: Bulleted list of specific quantitative metrics, entities, labels, percentages, dates, or relationships shown.
4. STRUCTURED_MARKDOWN: If this is a table or structured chart, reconstruct the exact data as a GitHub-flavored Markdown table. If not applicable, write "N/A".
5. SEARCH_KEYWORDS: 5-8 comma-separated search keywords that should match queries searching for this visual data.
"""


class VisionLLMClient:
    """Client for multimodal LLM vision analysis."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider or settings.VISION_MODEL_PROVIDER
        self.model_name = model_name or settings.VISION_MODEL_NAME
        self.api_key = api_key or (
            settings.OPENAI_API_KEY if self.provider == "openai" else settings.ANTHROPIC_API_KEY
        )
        
        # Check if we should activate mock mode
        if self.provider != "mock" and not self.api_key:
            logger.warning(
                f"No API key detected for {self.provider}. Falling back to VisionLLMClient in 'mock' mode. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env to enable real vision inference."
            )
            self.provider = "mock"

    @staticmethod
    def encode_image(image_path: str | Path) -> Tuple[str, str]:
        """Encodes an image to base64 string and determines its MIME type."""
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

        with open(path, "rb") as image_file:
            encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_str, mime_type

    def analyze_image(
        self,
        image_path: str | Path,
        context_hint: str = "",
        is_table: bool = False,
    ) -> Dict[str, Any]:
        """
        Submits an image to the Vision LLM and parses the structured response.
        Returns a dictionary containing element_type, description, markdown_table, and keywords.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")

        prompt = f"Analyze this image. Context from surrounding page text: '{context_hint}'."
        if is_table:
            prompt += " Note: This element has been identified as a table. Please reconstruct the Markdown table accurately."

        if self.provider == "openai":
            return self._analyze_openai(path, prompt)
        elif self.provider == "anthropic":
            return self._analyze_anthropic(path, prompt)
        else:
            return self._analyze_mock(path, is_table)

    def _analyze_openai(self, image_path: Path, prompt: str) -> Dict[str, Any]:
        """Inference using OpenAI Vision models (e.g. gpt-4o-mini)."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            base64_img, mime_type = self.encode_image(image_path)
            data_url = f"data:{mime_type};base64,{base64_img}"

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_VISION_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ],
                    },
                ],
                max_tokens=settings.VISION_MAX_TOKENS,
                temperature=settings.VISION_TEMPERATURE,
            )
            raw_text = response.choices[0].message.content or ""
            return self._parse_vision_response(raw_text)

        except Exception as e:
            logger.error(f"OpenAI Vision inference failed: {e}. Falling back to mock description.")
            return self._analyze_mock(image_path, is_table=False)

    def _analyze_anthropic(self, image_path: Path, prompt: str) -> Dict[str, Any]:
        """Inference using Anthropic Claude 3.5 Sonnet Vision."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            base64_img, mime_type = self.encode_image(image_path)

            response = client.messages.create(
                model=self.model_name or "claude-3-5-sonnet-20241022",
                max_tokens=settings.VISION_MAX_TOKENS,
                system=SYSTEM_VISION_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_img,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            raw_text = response.content[0].text if response.content else ""
            return self._parse_vision_response(raw_text)

        except Exception as e:
            logger.error(f"Anthropic Vision inference failed: {e}. Falling back to mock description.")
            return self._analyze_mock(image_path, is_table=False)

    def _analyze_mock(self, image_path: Path, is_table: bool = False) -> Dict[str, Any]:
        """Offline deterministic mock analyzer for local verification and testing."""
        stem = image_path.stem
        if is_table:
            elem_type = VisualElementType.TABLE
            desc = (
                f"[Table Analysis: {stem}] Extracted structured tabular data from document. "
                "Contains quantitative metrics, columns, and rows aligned with surrounding page context."
            )
            md_table = (
                "| Metric | Q1 | Q2 | Target |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| Performance Score | 92.4% | 96.1% | 95.0% |\n"
                "| Latency (ms) | 120ms | 85ms | 100ms |"
            )
        else:
            elem_type = VisualElementType.IMAGE
            desc = (
                f"[Visual Element: {stem}] High-resolution diagram/chart illustrating system workflow, "
                "components interaction, and quantitative distribution relevant to the document section."
            )
            md_table = None

        return {
            "element_type": elem_type,
            "description": desc,
            "markdown_table": md_table,
            "keywords": ["system architecture", "multimodal", "metrics", "data visualization"],
            "raw_text": desc,
        }

    def _parse_vision_response(self, text: str) -> Dict[str, Any]:
        """Parses structured text output from the Vision LLM."""
        parsed: Dict[str, Any] = {
            "element_type": VisualElementType.UNKNOWN,
            "description": "",
            "markdown_table": None,
            "keywords": [],
            "raw_text": text,
        }

        elem_type_str = ""
        summary = ""
        key_points = []
        md_table_lines = []
        collecting_table = False

        for line in text.splitlines():
            line_str = line.strip()
            if line_str.startswith("1. ELEMENT_TYPE:") or line_str.startswith("ELEMENT_TYPE:"):
                elem_type_str = line_str.split(":", 1)[1].strip().lower()
            elif line_str.startswith("2. SUMMARY:") or line_str.startswith("SUMMARY:"):
                collecting_table = False
                summary = line_str.split(":", 1)[1].strip()
            elif line_str.startswith("3. KEY_DATA_POINTS:") or line_str.startswith("KEY_DATA_POINTS:"):
                collecting_table = False
            elif line_str.startswith("4. STRUCTURED_MARKDOWN:") or line_str.startswith("STRUCTURED_MARKDOWN:"):
                collecting_table = True
            elif line_str.startswith("5. SEARCH_KEYWORDS:") or line_str.startswith("SEARCH_KEYWORDS:"):
                collecting_table = False
                kw_str = line_str.split(":", 1)[1].strip()
                parsed["keywords"] = [k.strip() for k in kw_str.split(",") if k.strip()]
            elif collecting_table:
                if line_str and not line_str.startswith("5.") and line_str != "N/A":
                    md_table_lines.append(line)
            elif line_str.startswith("-") or line_str.startswith("*"):
                key_points.append(line_str)
            elif summary and not collecting_table:
                # Append multi-line summary
                summary += " " + line_str

        # Resolve element type enum
        if "table" in elem_type_str:
            parsed["element_type"] = VisualElementType.TABLE
        elif "chart" in elem_type_str:
            parsed["element_type"] = VisualElementType.CHART
        elif "diagram" in elem_type_str:
            parsed["element_type"] = VisualElementType.DIAGRAM
        elif "infographic" in elem_type_str:
            parsed["element_type"] = VisualElementType.INFOGRAPHIC
        else:
            parsed["element_type"] = VisualElementType.IMAGE

        # Combine description
        full_desc = summary.strip()
        if key_points:
            full_desc += "\nKey Data Points:\n" + "\n".join(key_points)
        parsed["description"] = full_desc if full_desc else text.strip()

        # Extract markdown table
        if md_table_lines:
            joined_md = "\n".join(md_table_lines).strip()
            if "|" in joined_md:
                parsed["markdown_table"] = joined_md

        return parsed

