from generic import load_data, plot_data, np
from clustering import standard_kmeans


def split_data(X, y, ratio=0.9):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_point = int(len(X) * ratio)

    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


class MinDistanceClassifier:
    def __init__(self):
        self.representatives = None
        self.rep_labels = None

    def train(self, X, y, centers_per_class=1):
        classes = np.unique(y)
        all_reps = []
        all_labels = []

        for cls in classes:
            cls_data = X[y == cls]
            if centers_per_class == 1:
                # Jeden střed (těžiště)
                centroids = cls_data.mean(axis=0).reshape(1, -1)
            else:
                # Více středů (použijeme dříve napsanou funkci standard_kmeans)
                centroids, _ = standard_kmeans(cls_data, k=centers_per_class)

            all_reps.append(centroids)
            # Každému reprezentantu přiřadíme label třídy, ke které patří
            all_labels.append(np.full(len(centroids), cls))

        # Převedeme na velké matice pro hromadné výpočty
        self.representatives = np.vstack(all_reps)  # Matice (Počet_zástupců x 2)
        self.rep_labels = np.concatenate(all_labels)  # Vektor (Počet_zástupců,)

    def predict(self, X_new):
        """
        X_new může být jeden bod [x, y] nebo matice bodů (N x 2)
        """
        # Zajistíme, aby vstup byla matice (i pro jeden bod)
        X_new = np.atleast_2d(X_new)

        # MAGIE BROADCASTINGU:
        # 1. Rozšíříme dimenze, aby NumPy mohl vypočítat vzdálenosti "každý s každým"
        # X_new[:, np.newaxis] má tvar (N, 1, 2)
        # self.representatives má tvar (R, 2)
        # Výsledek rozdílu bude (N, R, 2)
        diff = X_new[:, np.newaxis, :] - self.representatives

        # 2. Vypočítáme Euklidovskou vzdálenost (normu) přes poslední osu (souřadnice)
        # dists bude mít tvar (N, R) -> vzdálenost každého z N bodů ke každému z R zástupců
        dists = np.linalg.norm(diff, axis=2)

        # 3. Pro každý z N bodů najdeme index nejbližšího zástupce
        closest_rep_indices = np.argmin(dists, axis=1)

        # 4. Vrátíme labely tříd odpovídající těmto zástupcům
        return self.rep_labels[closest_rep_indices]


if __name__ == '__main__':
    data_list = load_data("data_kla.txt")
    data = np.array(data_list, dtype=np.float32)
    X_all = data[:,:2]
    y_all = data[:,2].astype(int)
    X_train, y_train, X_test, y_test = split_data(X_all, y_all)

    plot_data(X_train, y_train)
    plot_data(X_test, y_test)

    mdc = MinDistanceClassifier()
    mdc.train(X_train, y_train, 3)
    results = mdc.predict(X_test) == y_test
    print(results)
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
    print(mdc.representatives)
    print(mdc.rep_labels)
