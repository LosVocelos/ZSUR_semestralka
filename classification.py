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
                # Více středů (použijeme dříve funkci standard_kmeans)
                centroids, _ = standard_kmeans(cls_data, k=centers_per_class)

            all_reps.append(centroids)
            # Každému reprezentantu přiřadíme label třídy, ke které patří
            all_labels.append(np.full(len(centroids), cls))

        # Převedeme na velké matice pro hromadné výpočty
        self.representatives = np.vstack(all_reps)  # Matice (Počet_zástupců x 2)
        self.rep_labels = np.concatenate(all_labels)  # Vektor (Počet_zástupců,)

    def predict(self, point):
        dists = np.linalg.norm(self.representatives - point, axis=1)
        # najdeme index nejbližšího etalonu
        closest_rep_index = np.argmin(dists)
        # A vrátíme label třídy odpovídající tomuto etalonu
        return self.rep_labels[closest_rep_index]


class KNNClassifier:
    def __init__(self, k_neighbors=1):
        self.k = k_neighbors

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def predict_vote(self, point):
        dists = np.linalg.norm(self.X_train - point, axis=1)
        # Najdeme indexy k nejbližších sousedů
        nearest_indices = np.argsort(dists)[:self.k]
        nearest_labels = self.y_train[nearest_indices]
        # Hlasování (majority vote)
        counts = np.bincount(nearest_labels.astype(int))
        return np.argmax(counts)

    def predict_dist(self, point):
        class_averages = {}

        for cls in self.classes:
            # 1. Vybereme body patřící do aktuální třídy a spočítáme vzdálenosti
            cls_data = self.X_train[self.y_train == cls]
            distances = np.linalg.norm(cls_data - point, axis=1)

            # 2. Vybereme k nejmenších vzdáleností
            # Pokud má třída méně bodů než k, vezmeme všechny dostupné
            actual_k = min(self.k, len(distances))
            k_nearest_distances = np.partition(distances, actual_k - 1)[:actual_k]

            # 4. Spočítáme průměrnou vzdálenost k těmto k sousedům
            avg_dist = np.mean(k_nearest_distances)
            class_averages[cls] = avg_dist

        # 5. Vybereme třídu s nejmenší průměrnou vzdáleností
        return min(class_averages, key=class_averages.get)


if __name__ == '__main__':
    data_list = load_data("data_kla.txt")
    data = np.array(data_list, dtype=np.float32)
    X_all = data[:,:2]
    y_all = data[:,2].astype(int)
    X_train, y_train, X_test, y_test = split_data(X_all, y_all)

    plot_data(X_train, y_train)
    plot_data(X_test, y_test)

    # Klasifikátor podle minimální vzdálenosti
    mdc = MinDistanceClassifier()
    mdc.train(X_train, y_train, 3)
    results = [mdc.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")

    # Klasifikátor podle nejbližšího souseda
    knn = KNNClassifier(3)
    knn.train(X_train, y_train)
    results = [knn.predict_vote(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
    results = [knn.predict_dist(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
