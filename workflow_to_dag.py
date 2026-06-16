#!/usr/bin/env python3
"""
Visualize a YAML workflow as a directed acyclic graph (DAG).

Parses step references (${step_id.outputs.name}) to build the dependency
graph, then renders it using Graphviz dot or matplotlib.

Usage:
  python workflow_to_dag.py workflow.yaml                    # PNG via dot
  python workflow_to_dag.py workflow.yaml -o dag.pdf         # PDF via dot
  python workflow_to_dag.py workflow.yaml --engine matplotlib # fallback
  python workflow_to_dag.py workflow.yaml --dot              # print DOT source
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Parse workflow YAML → dependency graph
# ---------------------------------------------------------------------------

REF_PATTERN = re.compile(r'\$\{(\w+)\.outputs\.(\w+)\}')


def parse_workflow(yaml_path):
    """Parse a YAML workflow and return steps with dependencies.

    Returns list of dicts:
      {id, tool, params, outputs, depends_on: [(step_id, output_name), ...]}
    """
    with open(yaml_path) as f:
        wf = yaml.safe_load(f)

    steps = []
    for step in wf.get('steps', []):
        step_id = step['id']
        tool = step.get('tool', '?')
        outputs = step.get('outputs', {})

        # Find all ${step_id.outputs.name} references in params
        deps = set()
        _find_refs(step.get('params', {}), deps)

        steps.append({
            'id': step_id,
            'tool': tool,
            'outputs': outputs,
            'depends_on': sorted(deps),
        })

    return wf.get('name', 'Workflow'), steps


def _find_refs(obj, deps):
    """Recursively find ${step.outputs.name} references."""
    if isinstance(obj, str):
        for match in REF_PATTERN.finditer(obj):
            deps.add((match.group(1), match.group(2)))
    elif isinstance(obj, dict):
        for v in obj.values():
            _find_refs(v, deps)
    elif isinstance(obj, list):
        for item in obj:
            _find_refs(item, deps)


# ---------------------------------------------------------------------------
# Build DOT source
# ---------------------------------------------------------------------------

# Tool category → color
TOOL_COLORS = {
    'fetch': '#E3F2FD',      # light blue
    'load': '#FFF3E0',       # light orange
    'extract': '#E8F5E9',    # light green
    'match': '#FCE4EC',      # light pink
    'compute': '#F3E5F5',    # light purple
    'plot': '#FFF9C4',       # light yellow
}


def tool_color(tool_name):
    """Pick a fill color based on tool name prefix."""
    for prefix, color in TOOL_COLORS.items():
        if prefix in tool_name:
            return color
    return '#F5F5F5'  # default gray


def build_dot(name, steps, show_outputs=True):
    """Build Graphviz DOT source string."""
    lines = [
        'digraph workflow {',
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8, color="#666666"];',
        f'  label=<<B>{_escape_html(name)}</B>>;',
        '  labelloc=t;',
        '  fontname="Helvetica";',
        '  fontsize=13;',
        '  nodesep=0.4;',
        '  ranksep=0.5;',
        '',
    ]

    # Step nodes
    for i, step in enumerate(steps):
        color = tool_color(step['tool'])
        label = f"{i+1}. {step['id']}\\n({step['tool']})"
        lines.append(
            f'  "{step["id"]}" [label="{label}", fillcolor="{color}"];'
        )

    lines.append('')

    # Output file nodes (small ellipses)
    if show_outputs:
        output_nodes = set()
        for step in steps:
            for out_key, out_val in step['outputs'].items():
                node_id = f"{step['id']}__{out_key}"
                # Extract just the filename from the path
                filename = Path(str(out_val)).name
                if '${' in str(out_val):
                    # Template not resolved — use the key name
                    filename = out_key
                if node_id not in output_nodes:
                    output_nodes.add(node_id)
                    lines.append(
                        f'  "{node_id}" [label="{filename}", shape=note, '
                        f'style=filled, fillcolor="#ECEFF1", fontsize=8, '
                        f'width=0.3, height=0.2];'
                    )

        lines.append('')

        # Edges: step → output file
        for step in steps:
            for out_key in step['outputs']:
                node_id = f"{step['id']}__{out_key}"
                lines.append(f'  "{step["id"]}" -> "{node_id}" [arrowsize=0.6];')

        lines.append('')

        # Edges: output file → consuming step
        for step in steps:
            for dep_step_id, dep_out_key in step['depends_on']:
                node_id = f"{dep_step_id}__{dep_out_key}"
                lines.append(
                    f'  "{node_id}" -> "{step["id"]}" '
                    f'[arrowsize=0.6];'
                )
    else:
        # Simplified: direct step → step edges
        for step in steps:
            seen = set()
            for dep_step_id, dep_out_key in step['depends_on']:
                if dep_step_id not in seen:
                    seen.add(dep_step_id)
                    lines.append(
                        f'  "{dep_step_id}" -> "{step["id"]}" '
                        f'[label="{dep_out_key}"];'
                    )

    lines.append('}')
    return '\n'.join(lines)


def _escape_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_dot(dot_src, output_path, fmt=None):
    """Render DOT source to a file using Graphviz dot."""
    if fmt is None:
        fmt = Path(output_path).suffix.lstrip('.') or 'png'

    try:
        result = subprocess.run(
            ['dot', f'-T{fmt}', '-o', str(output_path)],
            input=dot_src,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Graphviz error: {result.stderr}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print("Graphviz 'dot' not found. Use --dot to print source, "
              "or --engine matplotlib.", file=sys.stderr)
        return False


def render_matplotlib(name, steps, output_path):
    """Fallback renderer using networkx + matplotlib."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    # Add step nodes
    for step in steps:
        G.add_node(step['id'], tool=step['tool'])

    # Add edges
    for step in steps:
        seen = set()
        for dep_step_id, _ in step['depends_on']:
            if dep_step_id not in seen:
                seen.add(dep_step_id)
                G.add_edge(dep_step_id, step['id'])

    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(steps) * 0.8)))

    # Use graphviz layout if available, otherwise spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception:
        try:
            pos = nx.planar_layout(G)
        except Exception:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    colors = [tool_color(G.nodes[n].get('tool', '')) for n in G.nodes]
    labels = {n: f"{i+1}. {n}\n({G.nodes[n].get('tool', '?')})"
              for i, n in enumerate(G.nodes)}

    nx.draw(G, pos, ax=ax, labels=labels, node_color=colors,
            node_size=3000, font_size=7, arrows=True,
            edge_color='#999999', arrowsize=15,
            node_shape='s', linewidths=1, edgecolors='#CCCCCC')

    ax.set_title(name, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a YAML workflow as a DAG")
    parser.add_argument("workflow", help="Path to workflow YAML file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (default: <workflow>_dag.png)")
    parser.add_argument("--engine", choices=["dot", "matplotlib"],
                        default="dot",
                        help="Rendering engine (default: dot)")
    parser.add_argument("--dot", action="store_true",
                        help="Print DOT source to stdout (no rendering)")
    parser.add_argument("--simplified", action="store_true",
                        help="Hide intermediate file nodes, show only "
                             "step-to-step edges")
    args = parser.parse_args()

    yaml_path = Path(args.workflow)
    if not yaml_path.exists():
        print(f"File not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    name, steps = parse_workflow(yaml_path)

    if not steps:
        print("No steps found in workflow.", file=sys.stderr)
        sys.exit(1)

    print(f"Workflow: {name}")
    print(f"Steps: {len(steps)}")
    for s in steps:
        deps = ', '.join(f'{d[0]}.{d[1]}' for d in s['depends_on'])
        print(f"  {s['id']} ({s['tool']}) "
              f"{'← ' + deps if deps else ''}")

    show_outputs = not args.simplified
    dot_src = build_dot(name, steps, show_outputs=show_outputs)

    if args.dot:
        print(dot_src)
        return

    output = args.output or str(yaml_path.with_name(yaml_path.stem + '_dag.png'))

    if args.engine == 'dot':
        if render_dot(dot_src, output):
            print(f"\nSaved: {output}")
        else:
            print("Falling back to matplotlib...", file=sys.stderr)
            if render_matplotlib(name, steps, output):
                print(f"\nSaved: {output}")
    else:
        if render_matplotlib(name, steps, output):
            print(f"\nSaved: {output}")


if __name__ == '__main__':
    main()
