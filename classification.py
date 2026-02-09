from generic import load_data, plot_data, np
from clustering import standard_kmeans


def split_data(X, y, ratio=0.9):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_point = int(len(X) * ratio)

    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


class MinDistanceClassifier:
    def __init__(self, centers_per_class=1):
        self.representatives = None
        self.rep_labels = None
        self.n_centers = centers_per_class

    def train(self, X, y):
        classes = np.unique(y)
        all_reps = []
        all_labels = []

        for cls in classes:
            cls_data = X[y == cls]
            if self.n_centers == 1:
                # Jeden střed (těžiště)
                centroids = cls_data.mean(axis=0).reshape(1, -1)
            else:
                # Více středů (použijeme dříve funkci standard_kmeans)
                centroids, _ = standard_kmeans(cls_data, k=self.n_centers)

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
        self.X_train = None
        self.y_train = None
        self.classes = None
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


class BayesClassifier:
    def __init__(self):
        self.classes = None
        self.mu = None          # (R, 2)
        self.inv_sigma = None   # (R, 2, 2)
        self.det_sigma = None   # (R,)
        self.priors = None      # (R,)
        self.norm_const = None  # (R,)

    def train(self, X, y):
        self.classes = np.unique(y)
        R = len(self.classes)
        n_total = len(X)
        d = X.shape[1]

        # Inicializace polí pro parametry
        self.mu = np.zeros((R, d))
        self.inv_sigma = np.zeros((R, d, d))
        self.det_sigma = np.zeros(R)
        self.priors = np.zeros(R)

        for i, cls in enumerate(self.classes):
            cls_data = X[y == cls]

            # Výpočty pro konkrétní třídu
            self.mu[i] = np.mean(cls_data, axis=0)
            sigma = np.cov(cls_data, rowvar=False)

            # Regularizace (pro jistotu, aby determinant nebyl 0)
            sigma += np.eye(d) * 1e-6

            self.inv_sigma[i] = np.linalg.inv(sigma)
            self.det_sigma[i] = np.linalg.det(sigma)
            self.priors[i] = len(cls_data) / n_total

        # Předvýpočet normalizační konstanty: 1 / (2*pi * sqrt(det))
        # Pro 2D je to (2*pi)**(2/2) = 2*pi
        self.norm_const = 1.0 / (2.0 * np.pi * np.sqrt(self.det_sigma))

    def predict(self, point):
        point = np.array(point)  # Vektor (2,)

        # 1. Výpočet rozdílů (point - mu) pro všechny třídy najednou
        # point: (2,), self.mu: (R, 2) -> diff: (R, 2)
        diff = point - self.mu

        # 2. Výpočet exponentu pro všechny třídy najednou
        # Potřebujeme (diff[i]^T @ inv_sigma[i] @ diff[i]) pro každé i.
        # K tomu je v NumPy nejlepší funkce np.einsum (Einsteinova sumační konvence)
        # 'ki,kij,kj->k' znamená:
        # k = index třídy, i a j = souřadnice x,y
        # Pro každé k vynásob diff[k,i] * inv_sigma[k,i,j] * diff[k,j] a sečti přes i,j
        exponent_term = np.einsum('ki,kij,kj->k', diff, self.inv_sigma, diff)

        # 3. Finální výpočet f(x|omega) * P(omega) pro všechny třídy
        likelihoods = self.norm_const * np.exp(-0.5 * exponent_term)
        posteriors = likelihoods * self.priors

        # 4. Vrátíme label třídy s nejvyšší hodnotou
        return self.classes[np.argmax(posteriors)]


if __name__ == '__main__':
    data_list = load_data("data_kla.txt")
    data = np.array(data_list, dtype=np.float32)
    X_all = data[:,:2]
    y_all = data[:,2].astype(int)
    X_train, y_train, X_test, y_test = split_data(X_all, y_all)

    plot_data(X_train, y_train)
    plot_data(X_test, y_test)

    # Klasifikátor podle minimální vzdálenosti
    print("Minimal distance Classifier")
    mdc = MinDistanceClassifier(3)
    mdc.train(X_train, y_train)
    results = [mdc.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")

    # Klasifikátor podle nejbližšího souseda
    print("k-Nearest Neighbours Classifier")
    knn = KNNClassifier(3)
    knn.train(X_train, y_train)
    results = [knn.predict_vote(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy (vote): {sum(results)/len(results) *100:.2f}%")
    results = [knn.predict_dist(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy (dist): {sum(results)/len(results) *100:.2f}%")

    # Bayesův klasifikátor
    print("Bayes Classifier")
    bayes = BayesClassifier()
    bayes.train(X_train, y_train)
    results = [bayes.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
