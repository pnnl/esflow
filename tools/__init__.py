"""
ESFlow Analysis Tools

Modular tools for analyzing Earth System Model output.
Tools are designed to be chained together via YAML workflow files.

Usage:
    python run_workflow.py workflows/my_workflow.yaml

Tool Categories:
    - loaders: Load model and observation data
    - matchers: Match gauges to grid, trace rivers, extract regions
    - extractors: Extract time series, fields, and profiles
    - analyzers: Compute metrics, climatologies, and differences
    - plotters: Create publication-ready visualizations
"""

__version__ = '0.1.0'
