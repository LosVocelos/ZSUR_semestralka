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


def plot_data(data:list[tuple[float, float]] | np.ndarray, labels:list[int] | np.ndarray, title:str) -> None:
    """
    Vykreslení dat do plotu
    """
    xpoints = np.array([t[0] for t in data])
    ypoints = np.array([t[1] for t in data])

    plt.scatter(xpoints, ypoints, c=labels, edgecolors='k', cmap='rainbow', s=20)
    plt.title(title)
    plt.show()


def plot_boundaries(classifier, X, y, title):
    h = 0.1  # Krok mřížky
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predikce pro celou mřížku
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([classifier.predict(p) for p in grid_points])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='rainbow', s=20)
    plt.title(title)
    plt.show()
