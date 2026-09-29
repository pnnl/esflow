#!/usr/bin/env python3
"""
Generate tool_catalog.yaml from tool specifications.

Auto-discovers all tools in category subdirectories, imports them
to populate TOOL_REGISTRY, then generates the catalog.
"""

import importlib
import sys
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))

# Category directories to scan
TOOL_CATEGORIES = ['fetchers', 'loaders', 'matchers', 'extractors', 'analyzers', 'plotters', 'validators']


def discover_and_import():
    """Auto-discover and import all tool modules."""
    print("Discovering tools...")
    for category in TOOL_CATEGORIES:
        category_dir = tools_dir / category
        if not category_dir.exists():
            continue
        for py_file in category_dir.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
            module_name = f"{category}.{py_file.stem}"
            try:
                importlib.import_module(module_name)
            except Exception as e:
                print(f"  Warning: Failed to import {module_name}: {e}")


def generate_catalog():
    """Generate catalog from TOOL_REGISTRY."""
    import yaml
    from core.base import TOOL_REGISTRY

    discover_and_import()
    print(f"Registered {len(TOOL_REGISTRY)} tools")

    tools_list = []
    for name, spec in TOOL_REGISTRY.items():
        entry = spec.to_catalog_dict()
        # Infer category from file location
        for cat in TOOL_CATEGORIES:
            cat_dir = tools_dir / cat
            if (cat_dir / f"{name}.py").exists():
                entry['category'] = cat
                entry['path'] = f"{cat}/{name}.py"
                break
        tools_list.append(entry)

    catalog = {
        'version': '3.0',
        'description': 'ESMFlow v3 Analysis Tools',
        'tools': tools_list,
    }

    return catalog


def main():
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description='Generate tool catalog')
    parser.add_argument('--output', '-o', default='tool_catalog_generated.yaml',
                        help='Output file path')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite tool_catalog.yaml directly')
    args = parser.parse_args()

    catalog = generate_catalog()

    output_path = args.output
    if args.overwrite:
        output_path = tools_dir / 'tool_catalog.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nGenerated catalog with {len(catalog['tools'])} tools")
    print(f"Written to: {output_path}")

    # Summary
    for tool in catalog['tools']:
        n_inputs = len(tool.get('inputs', {}))
        n_outputs = len(tool.get('outputs', {}))
        cat = tool.get('category', '?')
        print(f"  [{cat}] {tool['name']} ({n_inputs} params, {n_outputs} outputs)")


if __name__ == '__main__':
    main()
