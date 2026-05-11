"""Base class for ingestible parameter objects."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator


class IngestibleParams(BaseModel):
    """Base class for derived runtime objects built from sanitized config sources."""
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True, arbitrary_types_allowed=True)

    @model_validator(mode='before')
    @classmethod
    def warn_extra(cls, data):
        if isinstance(data, dict):
            known = set(cls.model_fields.keys())
            extra = set(data.keys()) - known
            print(f"WARNING: Fields: '{extra}' not defined in {cls.__name__}, left unvalidated and directly passed to the runtime object. If this is intentional, ignore this warning.")
        return data

    @classmethod
    def from_sources(cls, *sources: BaseModel | dict[str, Any], **overrides: Any) -> Self:
        unified_data: dict[str, Any] = {}
        for source in sources:
            data = source.model_dump(mode="python") if isinstance(source, BaseModel) else source
            unified_data.update(data)
        unified_data.update(overrides)
        return cls.model_validate(unified_data)
