from matplotlib.axes import Axes


def despine(ax: Axes) -> None:
    """Hide the top and right axes spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
