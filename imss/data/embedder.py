"""Zhipu embedding-3 via OpenAI-compatible API."""

from __future__ import annotations

from openai import OpenAI

from imss.config import IMSSSettings


class ZhipuEmbedder:
    """Synchronous embedder using Zhipu embedding-3.

    Uses sync client because embeddings are typically batch-processed
    during data seeding, not in the hot simulation loop.
    """

    def __init__(self, settings: IMSSSettings | None = None):
        if settings is None:
            settings = IMSSSettings()
        self._client = OpenAI(
            api_key=settings.glm_api_key,
            base_url=settings.glm_base_url,
        )
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimension

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns vector of configured dimension."""
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimension,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one API call."""
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimension,
        )
        return [d.embedding for d in response.data]
