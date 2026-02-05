from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_data(filepath:str) -> list[tuple[float, float, int]]:
    """
    Načtení dat ze souboru
    """
    data: list[tuple[float, float, int]] = []

    with Path(filepath).open("r") as file:
        for line in file.readlines():
            x, y, *g = line.strip().split(",")
            data.append((float(x), float(y), int(*g)))

    return data


def plot_data(data:list[tuple[float, float]], labels:list[int]) -> None:
    """
    Vykreslení dat do plotu
    """
    xpoints = np.array([t[0] for t in data])
    ypoints = np.array([t[1] for t in data])
    colors = np.array(["C" + str(l) for l in labels])

    plt.scatter(xpoints, ypoints, marker=".", c=colors)
    plt.show()
