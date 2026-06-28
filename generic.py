from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


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


def plot_clusters(data: np.ndarray, labels: np.ndarray, title: str, centroids: np.ndarray = None) -> None:
    """
    Vykreslení dat s barevným odlišením shluků a volitelným zobrazením centroidů.
    """
    plt.figure(figsize=(8, 6))

    # Vykreslení samotných bodů
    plt.scatter(data[:, 0], data[:, 1], c=labels, edgecolors='k', cmap='rainbow', s=20)

    # Pokud jsou zadány centroidy, vykreslíme je výrazně do grafu
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroidy')
        plt.legend()

    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_boundaries(classifier, X, y, title):
    """
    Vykreslení dat a hranic rozhodování příslušného algoritmu
    """
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
