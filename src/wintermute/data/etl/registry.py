"""registry.py — Source plugin registration via @register_source decorator."""

from __future__ import annotations

from typing import Any

from wintermute.data.etl.base import DataSource


class SourceRegistry:
    """Singleton registry for data source plugins."""

    _sources: dict[str, type[DataSource]] = {}

    @classmethod
    def register(cls, name: str, source_cls: type[DataSource]) -> None:
        """Register a source class under the given name."""
        if name in cls._sources:
            raise ValueError(
                f"Source '{name}' is already registered "
                f"(existing: {cls._sources[name].__name__})"
            )
        cls._sources[name] = source_cls

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> DataSource:
        """Instantiate a registered source by name."""
        if name not in cls._sources:
            available = ", ".join(sorted(cls._sources.keys()))
            raise KeyError(
                f"Unknown source '{name}'. Available: [{available}]"
            )
        return cls._sources[name](config=config)

    @classmethod
    def available(cls) -> list[str]:
        """List all registered source names."""
        return sorted(cls._sources.keys())

    @classmethod
    def get(cls, name: str) -> type[DataSource] | None:
        """Get a source class by name, or None."""
        return cls._sources.get(name)


def register_source(name: str):
    """Decorator to register a DataSource subclass.

    Usage::

        @register_source("my_source")
        class MySource(DataSource):
            ...
    """
    def decorator(cls: type[DataSource]) -> type[DataSource]:
        SourceRegistry.register(name, cls)
        return cls
    return decorator
