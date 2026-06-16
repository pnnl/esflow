#!/usr/bin/env python3
"""
Generate a data-flow diagram from an ESMFlow workflow YAML.

Parses the workflow, extracts tool steps, output files, and inter-step
references, then renders a DAG as PNG.

Rendering backends (tried in order):
  1. System graphviz (dot command)
  2. Python graphviz package
  3. Matplotlib fallback (no extra deps)

Usage:
    python visualize_workflow.py workflow.yaml
    python visualize_workflow.py workflow.yaml -o diagram.png
    python visualize_workflow.py workflow.yaml --no-settings
"""

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# ── Colour palettes ──────────────────────────────────────────────────────

CATEGORY_COLORS = {
    'fetchers':   '#D4E6F1',
    'loaders':    '#D5F5E3',
    'matchers':   '#FEF9E7',
    'extractors': '#FDEBD0',
    'analyzers':  '#E8DAEF',
    'plotters':   '#FADBD8',
}

FILE_COLORS = {
    'csv':    '#F9E79F',
    'nc':     '#AED6F1',
    'netcdf': '#AED6F1',
    'png':    '#ABEBC6',
}


# ── YAML parsing helpers ─────────────────────────────────────────────────

def parse_references(obj):
    """Extract all ${...} reference strings from a nested object."""
    refs = []
    if isinstance(obj, str):
        refs.extend(re.findall(r'\$\{([^}]+)\}', obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            refs.extend(parse_references(v))
    elif isinstance(obj, list):
        for v in obj:
            refs.extend(parse_references(v))
    return refs


def load_catalog(workflow_path):
    """Try to load tool_catalog.yaml for category info."""
    candidates = [
        Path(workflow_path).parent.parent / 'tools' / 'tool_catalog.yaml',
        Path(workflow_path).parent.parent.parent / 'tools' / 'tool_catalog.yaml',
        Path(__file__).parent / 'tools' / 'tool_catalog.yaml',
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                cat = yaml.safe_load(f)
            return {t['name']: t for t in cat.get('tools', [])}
    return {}


# ── DAG structure ─────────────────────────────────────────────────────────

def build_dag(workflow, catalog):
    """Parse a workflow into nodes and edges for visualisation.

    Returns:
        nodes: list of dicts  {id, label, kind, category, color}
        edges: list of dicts  {src, dst, label, style}
    """
    steps = workflow.get('steps', [])
    settings = workflow.get('settings', {})
    nodes = []
    edges = []

    # Settings node
    settings_keys = [k for k in settings if k != 'output_dir']
    if settings_keys:
        nodes.append({
            'id': 'settings',
            'label': 'Settings\n' + '\n'.join(settings_keys),
            'kind': 'settings',
            'category': 'settings',
            'color': '#E8E8E8',
        })

    for step in steps:
        step_id = step.get('id', 'unknown')
        tool_name = step.get('tool', 'unknown')
        outputs = step.get('outputs', {})

        # Tool step node
        cat_info = catalog.get(tool_name, {})
        category = cat_info.get('category', 'unknown')
        color = CATEGORY_COLORS.get(category, '#F2F2F2')

        nodes.append({
            'id': step_id,
            'label': f'{step_id}\n({tool_name})',
            'kind': 'tool',
            'category': category,
            'color': color,
        })

        # Output file nodes
        for out_key, out_filename in outputs.items():
            if isinstance(out_filename, str) and not out_filename.startswith('$'):
                node_id = f'{step_id}__{out_key}'
                ext = Path(out_filename).suffix.lstrip('.')
                file_color = FILE_COLORS.get(ext, '#F2F2F2')
                nodes.append({
                    'id': node_id,
                    'label': out_filename,
                    'kind': 'file',
                    'category': 'file',
                    'color': file_color,
                })
                edges.append({
                    'src': step_id,
                    'dst': node_id,
                    'label': '',
                    'style': 'dashed',
                })

        # Edges from references
        params = step.get('params', step.get('config', {}))
        refs = parse_references(params)
        for ref in refs:
            parts = ref.split('.')
            if parts[0] == 'settings':
                edges.append({
                    'src': 'settings',
                    'dst': step_id,
                    'label': parts[-1],
                    'style': 'dotted',
                })
            elif len(parts) >= 3 and parts[1] == 'outputs':
                source_step = parts[0]
                source_key = parts[2]
                source_node = f'{source_step}__{source_key}'
                edges.append({
                    'src': source_node,
                    'dst': step_id,
                    'label': '',
                    'style': 'solid',
                })

    return nodes, edges


# ── Graphviz backend ──────────────────────────────────────────────────────

def build_dot(workflow, catalog, show_settings=True):
    """Build a Graphviz DOT string."""
    nodes, edges = build_dag(workflow, catalog)
    workflow_name = workflow.get('name', 'Workflow')

    lines = [
        'digraph workflow {',
        '    rankdir=TB;',
        '    fontname="Helvetica";',
        '    node [fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=8];',
        '    labelloc="t";',
        f'    label="{workflow_name}";',
        '    fontsize=14;',
        '',
    ]

    for n in nodes:
        nid = n['id']
        label = n['label'].replace('\n', '\\n')
        if n['kind'] == 'settings':
            if not show_settings:
                continue
            lines.append(f'    {nid} [label="{label}", shape=note, '
                         f'style=filled, fillcolor="{n["color"]}", fontsize=9];')
        elif n['kind'] == 'tool':
            lines.append(f'    {nid} [label="{label}", shape=box, '
                         f'style="filled,rounded", fillcolor="{n["color"]}"];')
        elif n['kind'] == 'file':
            lines.append(f'    {nid} [label="{label}", shape=ellipse, '
                         f'style=filled, fillcolor="{n["color"]}", fontsize=8];')

    lines.append('')

    for e in edges:
        if not show_settings and e['style'] == 'dotted':
            continue
        attrs = []
        if e['style'] == 'dotted':
            attrs.append('style=dotted')
            attrs.append('color="#AAAAAA"')
            if e['label']:
                attrs.append(f'label="{e["label"]}"')
                attrs.append('fontcolor="#999999"')
        elif e['style'] == 'dashed':
            attrs.append('style=dashed')
            attrs.append('color="#888888"')
        else:
            attrs.append('color="#2C3E50"')
            attrs.append('penwidth=1.5')

        attr_str = ', '.join(attrs)
        lines.append(f'    {e["src"]} -> {e["dst"]} [{attr_str}];')

    # Legend
    categories_used = sorted(set(
        n['category'] for n in nodes if n['kind'] == 'tool'
    ))
    if categories_used:
        lines.append('')
        lines.append('    subgraph cluster_legend {')
        lines.append('        label="Tool categories"; fontsize=9; '
                     'style=dashed; color="#CCCCCC";')
        for cat in categories_used:
            color = CATEGORY_COLORS.get(cat, '#F2F2F2')
            lines.append(f'        leg_{cat} [label="{cat}", shape=box, '
                         f'style="filled,rounded", fillcolor="{color}", fontsize=8];')
        if len(categories_used) > 1:
            chain = ' -> '.join(f'leg_{c}' for c in categories_used)
            lines.append(f'        {chain} [style=invis];')
        lines.append('    }')

    lines.append('}')
    return '\n'.join(lines)


def render_with_graphviz(dot_string, output_path):
    """Try system dot, then python-graphviz. Returns True on success."""
    dot_file = Path(output_path).with_suffix('.dot')
    dot_file.write_text(dot_string)

    # System graphviz
    if shutil.which('dot'):
        result = subprocess.run(
            ['dot', '-Tpng', '-Gdpi=150', str(dot_file), '-o', str(output_path)],
            capture_output=True, text=True,
        )
        dot_file.unlink(missing_ok=True)
        if result.returncode == 0:
            return True

    # Python graphviz package
    try:
        import graphviz as gv
        src = gv.Source(dot_string)
        out = src.render(
            filename=Path(output_path).stem,
            directory=str(Path(output_path).parent),
            format='png',
            cleanup=True,
        )
        dot_file.unlink(missing_ok=True)
        return True
    except ImportError:
        pass

    dot_file.unlink(missing_ok=True)
    return False


# ── Matplotlib fallback ───────────────────────────────────────────────────

def render_with_matplotlib(workflow, catalog, output_path, show_settings=True):
    """Render workflow DAG using matplotlib (no extra deps)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    nodes, edges = build_dag(workflow, catalog)

    if not show_settings:
        settings_ids = {n['id'] for n in nodes if n['kind'] == 'settings'}
        nodes = [n for n in nodes if n['kind'] != 'settings']
        edges = [e for e in edges if e['src'] not in settings_ids]

    if not nodes:
        return

    # Assign layers via topological order
    node_ids = [n['id'] for n in nodes]
    node_map = {n['id']: n for n in nodes}

    # Build adjacency for layer assignment
    children = defaultdict(list)
    parents = defaultdict(list)
    for e in edges:
        if e['src'] in node_map and e['dst'] in node_map:
            children[e['src']].append(e['dst'])
            parents[e['dst']].append(e['src'])

    # Assign layers: roots at top
    layers = {}
    def assign_layer(nid, depth=0):
        if nid in layers:
            layers[nid] = max(layers[nid], depth)
        else:
            layers[nid] = depth
        for child in children.get(nid, []):
            assign_layer(child, depth + 1)

    roots = [nid for nid in node_ids if not parents.get(nid)]
    for r in roots:
        assign_layer(r)
    # Assign unvisited nodes
    for nid in node_ids:
        if nid not in layers:
            layers[nid] = 0

    # Group nodes by layer
    layer_groups = defaultdict(list)
    for nid, layer in layers.items():
        layer_groups[layer].append(nid)

    max_layer = max(layers.values()) if layers else 0
    max_width = max(len(v) for v in layer_groups.values()) if layer_groups else 1

    # Compute positions
    positions = {}
    layer_height = 1.2
    node_spacing = 3.0

    for layer_idx in range(max_layer + 1):
        members = layer_groups[layer_idx]
        n_members = len(members)
        total_width = (n_members - 1) * node_spacing
        start_x = -total_width / 2
        for j, nid in enumerate(members):
            positions[nid] = (start_x + j * node_spacing, -layer_idx * layer_height)

    # Figure size — scale with content
    fig_width = max(max_width * node_spacing + 2, 6)
    fig_height = max((max_layer + 1) * layer_height + 1.5, 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect('equal')
    ax.axis('off')

    workflow_name = workflow.get('name', 'Workflow')
    ax.set_title(workflow_name, fontsize=14, fontweight='bold', pad=20)

    # Draw edges first (behind nodes)
    for e in edges:
        if e['src'] not in positions or e['dst'] not in positions:
            continue
        x0, y0 = positions[e['src']]
        x1, y1 = positions[e['dst']]

        style = '-'
        color = '#2C3E50'
        lw = 1.5
        alpha = 1.0
        if e['style'] == 'dashed':
            style = '--'
            color = '#888888'
            lw = 1.0
            alpha = 0.7
        elif e['style'] == 'dotted':
            style = ':'
            color = '#AAAAAA'
            lw = 0.8
            alpha = 0.5

        ax.annotate(
            '', xy=(x1, y1 + 0.35), xytext=(x0, y0 - 0.35),
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=lw, linestyle=style,
                connectionstyle='arc3,rad=0.1',
                shrinkA=5, shrinkB=5,
            ),
            alpha=alpha,
        )

    # Draw nodes
    for n in nodes:
        if n['id'] not in positions:
            continue
        x, y = positions[n['id']]

        if n['kind'] == 'tool':
            box_w, box_h = 2.4, 0.7
            box = mpatches.FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle='round,pad=0.1',
                facecolor=n['color'], edgecolor='#333333', linewidth=1.2,
            )
            ax.add_patch(box)
            # Two-line label: step_id bold, tool_name smaller
            parts = n['label'].split('\n')
            ax.text(x, y + 0.1, parts[0], ha='center', va='center',
                    fontsize=9, fontweight='bold')
            if len(parts) > 1:
                ax.text(x, y - 0.15, parts[1], ha='center', va='center',
                        fontsize=7, color='#555555')

        elif n['kind'] == 'file':
            box_w, box_h = 2.2, 0.45
            ellipse = mpatches.Ellipse(
                (x, y), box_w, box_h,
                facecolor=n['color'], edgecolor='#666666', linewidth=0.8,
            )
            ax.add_patch(ellipse)
            ax.text(x, y, n['label'], ha='center', va='center',
                    fontsize=7, style='italic')

        elif n['kind'] == 'settings':
            box_w, box_h = 2.0, 0.5
            box = mpatches.FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2), box_w, box_h,
                boxstyle='round,pad=0.05',
                facecolor=n['color'], edgecolor='#999999', linewidth=0.8,
            )
            ax.add_patch(box)
            ax.text(x, y, 'Settings', ha='center', va='center',
                    fontsize=8, color='#666666')

    # Legend
    categories_used = sorted(set(
        n['category'] for n in nodes if n['kind'] == 'tool'
    ))
    if categories_used:
        legend_patches = [
            mpatches.Patch(
                facecolor=CATEGORY_COLORS.get(c, '#F2F2F2'),
                edgecolor='#333333', label=c,
            )
            for c in categories_used
        ]
        ax.legend(
            handles=legend_patches, loc='lower right',
            fontsize=8, title='Tool categories', title_fontsize=9,
            framealpha=0.9,
        )

    # Adjust limits
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    mx, my = 1.8, 1.0
    ax.set_xlim(min(all_x) - mx, max(all_x) + mx)
    ax.set_ylim(min(all_y) - my, max(all_y) + my)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ── Public API ────────────────────────────────────────────────────────────

def visualize_workflow(workflow_path, output_path=None, show_settings=True):
    """Generate a data-flow diagram from a workflow YAML.

    Args:
        workflow_path: Path to the workflow YAML file.
        output_path: Output PNG path. Defaults to <output_dir>/workflow_diagram.png.
        show_settings: Whether to show the settings node and edges.

    Returns:
        Path to the generated diagram file.
    """
    workflow_path = Path(workflow_path)
    with open(workflow_path) as f:
        workflow = yaml.safe_load(f)

    if output_path is None:
        output_dir = Path(workflow.get('settings', {}).get('output_dir', '.'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'workflow_diagram.png'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(workflow_path)

    # Try graphviz first (better DAG layout)
    dot_string = build_dot(workflow, catalog, show_settings=show_settings)
    if render_with_graphviz(dot_string, output_path):
        print(f"  Workflow diagram: {output_path}")
        return str(output_path)

    # Fallback to matplotlib
    render_with_matplotlib(workflow, catalog, output_path,
                           show_settings=show_settings)
    print(f"  Workflow diagram: {output_path}  (matplotlib)")
    return str(output_path)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate a data-flow diagram from an ESMFlow workflow YAML',
    )
    parser.add_argument('workflow', help='Path to workflow YAML file')
    parser.add_argument('-o', '--output',
                        help='Output PNG path (default: <output_dir>/workflow_diagram.png)')
    parser.add_argument('--no-settings', action='store_true',
                        help='Hide the settings node and its edges')

    args = parser.parse_args()
    visualize_workflow(args.workflow, args.output,
                       show_settings=not args.no_settings)


if __name__ == '__main__':
    main()
