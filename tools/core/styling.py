"""
Plot styling utilities for ESM analysis tools.

Provides consistent color palettes and style presets for publication-ready figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any, Optional, List


# Color palettes for each style
COLOR_PALETTES = {
    'cream_ink': {
        'background': '#FFFFFF',
        'text': '#2C3E50',
        'primary': ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#9B59B6', '#F39C12'],
        'secondary': ['#34495E', '#C0392B', '#2980B9', '#1E8449', '#8E44AD', '#D68910'],
        'grid': '#D5DBDB',
        'accent': '#E74C3C',
    },
    'midnight_ink': {
        'background': '#1A1A2E',
        'text': '#EAEAEA',
        'primary': ['#00D9FF', '#FF6B6B', '#4ECDC4', '#FFE66D', '#C44569', '#6C5CE7'],
        'secondary': ['#00B4D8', '#EF476F', '#2EC4B6', '#FFD93D', '#A23B72', '#5B4FCF'],
        'grid': '#2D2D44',
        'accent': '#00D9FF',
    },
    'nature_retro': {
        'background': '#FFFFFF',
        'text': '#333333',
        'primary': ['#0077B6', '#E63946', '#2A9D8F', '#E9C46A', '#6D597A', '#BC6C25'],
        'secondary': ['#023E8A', '#D62828', '#264653', '#F4A261', '#5C4D7D', '#9A5518'],
        'grid': '#E5E5E5',
        'accent': '#E63946',
    },
    'science_minimal': {
        'background': '#FFFFFF',
        'text': '#000000',
        'primary': ['#000000', '#666666', '#999999', '#CCCCCC', '#333333', '#B3B3B3'],
        'secondary': ['#1A1A1A', '#4D4D4D', '#808080', '#B3B3B3', '#262626', '#A6A6A6'],
        'grid': '#E0E0E0',
        'accent': '#000000',
    },
    'presentation_bold': {
        'background': '#FFFFFF',
        'text': '#1A1A1A',
        'primary': ['#FF4136', '#0074D9', '#2ECC40', '#FF851B', '#B10DC9', '#FFDC00'],
        'secondary': ['#DC3030', '#0056A4', '#25A233', '#CC6A00', '#8B0A9E', '#CCA800'],
        'grid': '#DDDDDD',
        'accent': '#FF4136',
    },
}


# Style presets with full matplotlib configuration
STYLE_PRESETS = {
    'cream_ink': {
        'description': 'Clean, publication-ready style with white background',
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#2C3E50',
        'axes.labelcolor': '#2C3E50',
        'axes.titleweight': 'bold',
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#D5DBDB',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        'text.color': '#2C3E50',
        'xtick.color': '#2C3E50',
        'ytick.color': '#2C3E50',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.frameon': True,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#D5DBDB',
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'savefig.facecolor': '#FFFFFF',
        'savefig.edgecolor': 'none',
        'savefig.dpi': 300,
    },
    'midnight_ink': {
        'description': 'Dark mode style for digital display',
        'figure.facecolor': '#1A1A2E',
        'axes.facecolor': '#1A1A2E',
        'axes.edgecolor': '#EAEAEA',
        'axes.labelcolor': '#EAEAEA',
        'axes.titleweight': 'bold',
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#2D2D44',
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
        'text.color': '#EAEAEA',
        'xtick.color': '#EAEAEA',
        'ytick.color': '#EAEAEA',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.frameon': True,
        'legend.facecolor': '#1A1A2E',
        'legend.edgecolor': '#2D2D44',
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'savefig.facecolor': '#1A1A2E',
        'savefig.edgecolor': 'none',
        'savefig.dpi': 300,
    },
    'nature_retro': {
        'description': 'Classic Nature journal style',
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.titleweight': 'bold',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 1.0,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'savefig.facecolor': '#FFFFFF',
        'savefig.edgecolor': 'none',
        'savefig.dpi': 300,
    },
    'science_minimal': {
        'description': 'Grayscale-safe Science/PNAS style',
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#000000',
        'axes.labelcolor': '#000000',
        'axes.titleweight': 'bold',
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.linewidth': 0.8,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'text.color': '#000000',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'legend.frameon': True,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#000000',
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'savefig.facecolor': '#FFFFFF',
        'savefig.edgecolor': 'none',
        'savefig.dpi': 300,
    },
    'presentation_bold': {
        'description': 'Bold style for slides and posters',
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#1A1A1A',
        'axes.labelcolor': '#1A1A1A',
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'axes.linewidth': 2.0,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#DDDDDD',
        'grid.linestyle': '-',
        'grid.linewidth': 1.0,
        'grid.alpha': 0.5,
        'text.color': '#1A1A1A',
        'xtick.color': '#1A1A1A',
        'ytick.color': '#1A1A1A',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.frameon': True,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#DDDDDD',
        'legend.fontsize': 12,
        'lines.linewidth': 3.0,
        'lines.markersize': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'savefig.facecolor': '#FFFFFF',
        'savefig.edgecolor': 'none',
        'savefig.dpi': 150,
    },
}


def get_style_config(style_name: str = 'cream_ink') -> Dict[str, Any]:
    """
    Get style configuration dictionary.

    Args:
        style_name: Name of style preset

    Returns:
        Dictionary of matplotlib rcParams
    """
    if style_name not in STYLE_PRESETS:
        available = ', '.join(STYLE_PRESETS.keys())
        raise ValueError(f"Unknown style '{style_name}'. Available: {available}")

    return STYLE_PRESETS[style_name].copy()


def get_colors(style_name: str = 'cream_ink', n_colors: int = 6) -> List[str]:
    """
    Get color palette for a style.

    Args:
        style_name: Name of style preset
        n_colors: Number of colors to return

    Returns:
        List of hex color codes
    """
    if style_name not in COLOR_PALETTES:
        available = ', '.join(COLOR_PALETTES.keys())
        raise ValueError(f"Unknown style '{style_name}'. Available: {available}")

    palette = COLOR_PALETTES[style_name]
    colors = palette['primary'][:n_colors]

    # Extend with secondary colors if needed
    if n_colors > len(palette['primary']):
        colors.extend(palette['secondary'][:n_colors - len(colors)])

    return colors


def apply_style(style_name: str = 'cream_ink') -> None:
    """
    Apply style preset to matplotlib globally.

    Args:
        style_name: Name of style preset
    """
    config = get_style_config(style_name)

    # Remove description (not a valid rcParam)
    config.pop('description', None)

    # Apply to rcParams
    for key, value in config.items():
        try:
            plt.rcParams[key] = value
        except KeyError:
            # Skip invalid rcParams (e.g., custom keys)
            pass

    # Set color cycle
    colors = get_colors(style_name)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


def style_context(style_name: str = 'cream_ink'):
    """
    Context manager for temporarily applying a style.

    Usage:
        with style_context('midnight_ink'):
            plt.plot(...)
            plt.savefig(...)

    Args:
        style_name: Name of style preset

    Returns:
        Context manager
    """
    config = get_style_config(style_name)
    config.pop('description', None)

    # Add color cycle
    colors = get_colors(style_name)
    config['axes.prop_cycle'] = plt.cycler(color=colors)

    return plt.rc_context(config)


def create_figure(
    style_name: str = 'cream_ink',
    figsize: tuple = (10, 6),
    **kwargs
) -> tuple:
    """
    Create a figure with style applied.

    Args:
        style_name: Name of style preset
        figsize: Figure size (width, height) in inches
        **kwargs: Additional arguments to plt.subplots

    Returns:
        Tuple of (figure, axes)
    """
    with style_context(style_name):
        fig, ax = plt.subplots(figsize=figsize, **kwargs)

    return fig, ax


def style_colorbar(
    cbar,
    style_name: str = 'cream_ink',
    label: Optional[str] = None
) -> None:
    """
    Apply style to a colorbar.

    Args:
        cbar: Matplotlib colorbar object
        style_name: Name of style preset
        label: Optional label for colorbar
    """
    palette = COLOR_PALETTES.get(style_name, COLOR_PALETTES['cream_ink'])

    cbar.ax.tick_params(
        color=palette['text'],
        labelcolor=palette['text'],
        labelsize=9
    )
    cbar.outline.set_edgecolor(palette['text'])
    cbar.outline.set_linewidth(0.8)

    if label:
        cbar.set_label(label, color=palette['text'], fontsize=10)


def get_cmap_for_style(
    style_name: str = 'cream_ink',
    cmap_type: str = 'sequential'
) -> str:
    """
    Get recommended colormap for a style.

    Args:
        style_name: Name of style preset
        cmap_type: Type of colormap ('sequential', 'diverging', 'categorical')

    Returns:
        Colormap name
    """
    cmap_recommendations = {
        'cream_ink': {
            'sequential': 'YlOrRd',
            'diverging': 'RdBu_r',
            'categorical': None,  # Use palette colors
        },
        'midnight_ink': {
            'sequential': 'plasma',
            'diverging': 'coolwarm',
            'categorical': None,
        },
        'nature_retro': {
            'sequential': 'viridis',
            'diverging': 'RdYlBu_r',
            'categorical': None,
        },
        'science_minimal': {
            'sequential': 'Greys',
            'diverging': 'RdGy',
            'categorical': None,
        },
        'presentation_bold': {
            'sequential': 'hot',
            'diverging': 'seismic',
            'categorical': None,
        },
    }

    style_cmaps = cmap_recommendations.get(style_name, cmap_recommendations['cream_ink'])
    return style_cmaps.get(cmap_type, 'viridis')
