"""Wintermute ETL pipeline — modular, plugin-based data extraction."""

from wintermute.data.etl.base import DataSource, ExtractResult, PipelineResult, RawSample
from wintermute.data.etl.pipeline import Pipeline
from wintermute.data.etl.registry import SourceRegistry, register_source

__all__ = [
    "DataSource",
    "ExtractResult",
    "Pipeline",
    "PipelineResult",
    "RawSample",
    "SourceRegistry",
    "register_source",
]
