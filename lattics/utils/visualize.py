from lattics._units import convert_time
from lattics.recorders import Snapshot


def _pyplot():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError('Visualization needs matplotlib. Install it with: pip install "lattics[viz]"') from error
    return matplotlib, plt


def show_agents_and_substrates_2d(
    snapshot: Snapshot,
    times: tuple[tuple[float, str], ...] | str = 'all',
    color_by: str = None,
    cmap: str = 'viridis',
    color_mapping: dict = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    substrate: str = None,
    substrate_vmax: float = None,
    filename: str = None,
    show: bool = True,
) -> None:
    """Draws the agents (and a substrate field) of the frames of a snapshot
    recorded in a :class:`~lattics.Lattice2DSpace`.

    Parameters
    ----------
    snapshot : Snapshot
        The recorded frames
    times : tuple[tuple[float, str], ...] or str, optional
        The times of the frames to draw, for example ``((1, 'day'), (2, 'day'))``,
        or ``'all'``
    color_by : str, optional
        The attribute that sets the color of the agents
    cmap : str, optional
        The colormap for numeric attributes
    color_mapping : dict, optional
        The colors of the values of a non-numeric attribute
    vmin, vmax : float, optional
        The range of a numeric attribute
    substrate : str, optional
        The substrate whose field is drawn below the agents
    substrate_vmax : float, optional
        The upper limit of the color scale of the field
    filename : str, optional
        Save every figure as ``<filename>_<index>.png``
    show : bool, optional
        Show the figures
    """
    matplotlib, plt = _pyplot()
    if snapshot.dimensions is None:
        raise ValueError('The snapshot was not recorded in a lattice space.')
    frames = snapshot.frames if times == 'all' else [snapshot.frame_at(time) for time in times]
    colormap = matplotlib.colormaps[cmap]
    dims = snapshot.dimensions
    for counter, frame in enumerate(frames):
        fig, ax = plt.subplots()
        if substrate:
            # Transposed, so that site (i, j) is drawn at x = i, y = j like the agents
            field = frame.fields[substrate].T
            s_ax = ax.imshow(field, origin='lower', vmin=0, vmax=substrate_vmax, cmap='inferno')
            plt.colorbar(s_ax)
        for agent in frame.agents:
            if color_by:
                value = agent[color_by]
                if isinstance(value, str):
                    color = color_mapping[value]
                else:
                    color = colormap((value - vmin) / (vmax - vmin))
            else:
                color = 'blue'
            patch = plt.Circle(agent['space.position'], 0.4, edgecolor='black', facecolor=color, linewidth=0.2)
            ax.add_patch(patch)
        plt.title(f't = {convert_time(frame.time, "ms", "hour"):g} h')
        plt.axis('off')
        ax.set_xlim((-0.5, dims[0] - 0.5))
        ax.set_ylim((-0.5, dims[1] - 0.5))
        ax.set_aspect('equal')
        plt.tight_layout()
        if filename:
            plt.savefig(f'{filename}_{counter}.png', dpi=100)
        if show:
            plt.show()
        plt.close(fig)
