"""Base class for ingestible parameter objects."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class IngestibleParams(BaseModel):
    """Base class for derived runtime objects built from sanitized config sources."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def from_sources(cls, *sources: BaseModel | dict[str, Any], **overrides: Any) -> Self:
        unified_data: dict[str, Any] = {}
        for source in sources:
            data = source.model_dump(mode="python") if isinstance(source, BaseModel) else source
            unified_data.update(data)
        unified_data.update(overrides)
        return cls.model_validate(unified_data)
