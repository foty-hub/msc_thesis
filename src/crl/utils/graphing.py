import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def despine(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
