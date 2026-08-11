"""
ESMFlow Analysis Tools

Modular tools for analyzing Earth System Model output. Tools are discovered from
their decorated specifications and executed by ``common.workflow_runner``.

Tool Categories:
    - fetchers: Retrieve supported external datasets
    - loaders: Load model and observation data
    - matchers: Match gauges to grid, trace rivers, extract regions
    - extractors: Extract time series, fields, and profiles
    - analyzers: Compute metrics, climatologies, and differences
    - plotters: Create publication-ready visualizations
"""

__version__ = '0.1.0'
