from generic import load_data, plot_data, plot_boundaries, np, plt
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
    def __init__(self, k_neighbors=1, vote=True):
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.k = k_neighbors
        self.vote = vote

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

    def predict(self, point):
        return self.predict_vote(point) if self.vote else self.predict_dist(point)


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

        # 1. Výpočet rozdílů
        diff = point - self.mu

        # 2. Výpočet exponentu pro všechny třídy najednou
        # Potřebujeme (diff[i]^T @ inv_sigma[i] @ diff[i]) pro každé i.
        # 'ki,kij,kj->k' znamená:
        # k = index třídy, i a j = souřadnice x,y
        # Pro každé k vynásobíme diff[k,i] * inv_sigma[k,i,j] * diff[k,j] a sečti přes i,j
        exponent_term = np.einsum('ki,kij,kj->k', diff, self.inv_sigma, diff)

        # 3. Finální výpočet f(x|omega) * P(omega) pro všechny třídy
        likelihoods = self.norm_const * np.exp(-0.5 * exponent_term)
        posteriors = likelihoods * self.priors

        # 4. Vrátíme label třídy s nejvyšší hodnotou
        return self.classes[np.argmax(posteriors)]


class LinearDiscriminantClassifier:
    def __init__(self, rosenblatt=True, alpha=0.1, b=1.0, max_iters=10000):
        self.ros = rosenblatt
        self.alpha = alpha          # Konstanta učení
        self.b = b                  # Pásmo necitlivosti (pro upravenou metodu)
        self.max_iters = max_iters
        self.weights = None
        self.classes = None
        self.history = None

    def _prepare_data(self, X, y, target_class):
        """Rozšíření o bias a zrcadlení vektorů ostatních tříd."""
        X_aug = np.column_stack([X, np.ones(len(X))])
        # Zrcadlení: vektory cílové třídy zůstávají, ostatní * -1
        y_binary = np.where(y == target_class, 1, -1)
        return X_aug * y_binary[:, np.newaxis]

    def train_binary(self, X, y, target_class):
        Y = self._prepare_data(X, y, target_class)
        n_features = Y.shape[1]
        w = np.zeros(n_features)  # Počáteční váhy
        b = (0 if self.ros else self.b)

        iterations = 0
        for _ in range(self.max_iters):
            converged = True
            iterations += 1
            for y_vec in Y:
                # Výpočet skalárního součinu
                val = np.dot(w, y_vec)
                # Rozhodnutí o opravě
                if val <= b:
                    w = w + self.alpha * y_vec
                    converged = False

            if converged:
                break

        return w, iterations

    def train(self, X, y):
        self.classes = np.unique(y)
        self.weights = {}
        self.history = {}

        # Trénujeme jeden klasifikátor pro každou třídu (One-vs-Rest)
        for cls in self.classes:
            w, iters = self.train_binary(X, y, cls)
            self.weights[cls] = w
            self.history[cls] = iters
            print(f"Metoda {'rosenblatt' if self.ros else 'konst'}, třída {cls}, alfa {self.alpha}: {iters} iterací")

    def predict(self, point):
        p_aug = np.append(point, 1)
        scores = {cls: np.dot(self.weights[cls], p_aug) for cls in self.classes}
        return max(scores, key=scores.get)


class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        layer_sizes: Seznam velikostí vrstev, např. [2, 8, 3]
                     (2 vstupy, 8 neuronů ve skryté vrstvě, 3 výstupy)
        """
        self.sizes = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        # NAHODNÁ INICIALIZACE VAH (He / Xavier-like inicializace)
        # U neuronových sítí nemůžeme začít s nulami, protože by se všechny
        # skryté neurony učily identicky (problém symetrie).
        for i in range(len(self.sizes) - 1):
            w = np.random.randn(self.sizes[i], self.sizes[i + 1]) * np.sqrt(2.0 / self.sizes[i])
            b = np.zeros((1, self.sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivative(self, a):
        # a je již aktivovaná hodnota sigmoid(z)
        return a * (1.0 - a)

    def _softmax(self, z):
        # Stabilní softmax zabraňující přetečení exponentu
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, X):
        """Dopředný průchod sítí."""
        self.a = [X]  # Aktivace jednotlivých vrstev (a[0] je vstup X)
        self.z = []  # Hodnoty před aktivací (z = w*a + b)

        # Průchod skrytými vrstvami se Sigmoidou
        for i in range(len(self.weights) - 1):
            z_curr = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a_curr = self._sigmoid(z_curr)
            self.z.append(z_curr)
            self.a.append(a_curr)

        # Výstupní vrstva se Softmaxem (vhodné pro klasifikaci do více tříd)
        z_out = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        a_out = self._softmax(z_out)
        self.z.append(z_out)
        self.a.append(a_out)
        return a_out

    def backward(self, Y):
        """Zpětný průchod (Backpropagation) a aktualizace vah."""
        m = Y.shape[0]  # Počet vzorků v dávce (1 pro SGD, N pro Batch GD)
        dW_list = []
        db_list = []

        # Chyba na výstupu (pro kombinaci Softmax + Cross-Entropy je derivace velmi jednoduchá: a - Y)
        dZ = self.a[-1] - Y

        # Zpětná propagace od výstupu ke vstupu
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.a[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:
                # Výpočet chyby pro předchozí skrytou vrstvu
                dZ = np.dot(dZ, self.weights[i].T) * self._sigmoid_derivative(self.a[i])

        # Aktualizace vah a biasů
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * dW_list[i]
            self.biases[i] -= self.lr * db_list[i]

    def train(self, X, y, epochs, sgd=False):
        num_classes = self.sizes[-1]

        # Převedení labelů na One-Hot kódování
        Y = np.zeros((len(y), num_classes))
        Y[np.arange(len(y)), y.astype(int)] = 1

        loss_history = []

        for epoch in range(epochs):
            if sgd:
                # --- STOCHASTIC GRADIENT DESCENT (SGD) ---
                # Náhodné promíchání dat v každé epoše
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                epoch_loss = 0

                for idx in indices:
                    x_sample = X[idx:idx + 1]
                    y_sample = Y[idx:idx + 1]

                    out = self.forward(x_sample)
                    self.backward(y_sample)
                    epoch_loss -= np.sum(y_sample * np.log(out + 1e-15))

                loss_history.append(epoch_loss / len(X))

            else:
                # --- BATCH GRADIENT DESCENT ---
                # Výpočet pro všechna data najednou
                out = self.forward(X)
                self.backward(Y)
                loss = -np.mean(np.sum(Y * np.log(out + 1e-15), axis=1))
                loss_history.append(loss)

        return loss_history

    def predict(self, x):
        return self.forward(x)[0].argmax()


if __name__ == '__main__':
    data_list = load_data("data_kla.txt")
    data = np.array(data_list, dtype=np.float32)
    X_all = data[:,:2]
    y_all = data[:,2].astype(int)
    X_train, y_train, X_test, y_test = split_data(X_all, y_all)

    plot_data(X_train, y_train, "Training data")
    plot_data(X_test, y_test, "Testing data")

    # Klasifikátor podle minimální vzdálenosti
    print("Minimal distance Classifier")
    mdc = MinDistanceClassifier(3)
    mdc.train(X_train, y_train)
    results = [mdc.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
    plot_boundaries(mdc, X_all, y_all, "Minimal distance Classifier")

    # Klasifikátor podle nejbližšího souseda
    print("k-Nearest Neighbours Classifier")
    knn = KNNClassifier(3)
    knn.train(X_train, y_train)
    results = [knn.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy (vote): {sum(results)/len(results) *100:.2f}%")
    plot_boundaries(knn, X_all, y_all, "k-Nearest Neighbours Classifier (vote)")
    knn.vote = False
    results = [knn.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy (dist): {sum(results)/len(results) *100:.2f}%")
    plot_boundaries(knn, X_all, y_all, "k-Nearest Neighbours Classifier (dist)")

    # Bayesův klasifikátor
    print("Bayes Classifier")
    bay = BayesClassifier()
    bay.train(X_train, y_train)
    results = [bay.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
    plot_boundaries(bay, X_all, y_all, "Bayes Classifier")

    # Klasifikátor s lineárními diskriminačními funkcemi
    print("Linear Discriminant Classifier")
    lin = LinearDiscriminantClassifier(False, max_iters=1000)
    lin.train(X_train, y_train)
    results = [lin.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
    plot_boundaries(lin, X_all, y_all, "Linear Discriminant Classifier")

    # Klasifikátor s lineárními diskriminačními funkcemi
    print("Linear Discriminant Classifier (Rosenblatt)")
    lin = LinearDiscriminantClassifier(True, max_iters=1000)
    lin.train(X_train, y_train)
    results = [lin.predict(X) == y for X, y in zip(X_test, y_test)]
    print(f"Accuracy: {sum(results)/len(results) *100:.2f}%")
    plot_boundaries(lin, X_all, y_all, "Linear Discriminant Classifier (Rosenblatt)")

    # NN
    # Simulace ne-separabilních 2D dat (3 třídy)
    # np.random.seed(42)
    # X_train = np.vstack([
    #     np.random.normal([-1, -1], 0.8, (100, 2)),
    #     np.random.normal([2, 1], 0.8, (100, 2)),
    #     np.random.normal([-1, 2], 0.8, (100, 2))
    # ])
    # y_train = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # --- EXPERIMENT 1: Vliv topologie ---
    # Srovnáme lineární model [2, 3] vs. mělkou síť [2, 6, 3] vs. hlubší síť [2, 16, 8, 3]
    topologies = {
        "Lineární (bez skryté vrstvy) [2, 3]": [2, 3],
        "Jedna skrytá vrstva [2, 8, 3]": [2, 8, 3],
        "Dvě skryté vrstvy [2, 16, 8, 3]": [2, 16, 8, 3]
    }

    plt.figure(figsize=(10, 5))
    for name, topo in topologies.items():
        nn = SimpleNeuralNetwork(layer_sizes=topo, learning_rate=0.05)
        loss = nn.train(X_train, y_train, epochs=200, method='batch')
        plt.plot(loss, label=name)
    plt.title("Srovnání topologií (Batch GD)")
    plt.xlabel("Epocha")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # --- EXPERIMENT 2: Způsob trénování (SGD vs Batch GD) ---
    plt.figure(figsize=(10, 5))
    nn_batch = SimpleNeuralNetwork(layer_sizes=[2, 8, 3], learning_rate=0.05)
    loss_batch = nn_batch.train(X_train, y_train, epochs=300, method='batch')

    nn_sgd = SimpleNeuralNetwork(layer_sizes=[2, 8, 3], learning_rate=0.05)
    loss_sgd = nn_sgd.train(X_train, y_train, epochs=300, method='sgd')

    plt.plot(loss_batch, label="Batch Gradient Descent")
    plt.plot(loss_sgd, label="Stochastic Gradient Descent (SGD)")
    plt.title("Srovnání způsobů trénování")
    plt.xlabel("Epocha")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # --- EXPERIMENT 3: Volba konstanty učení (Learning Rate) ---
    lrs = [0.5, 0.05, 0.001]
    plt.figure(figsize=(10, 5))
    for lr in lrs:
        nn = SimpleNeuralNetwork(layer_sizes=[2, 8, 3], learning_rate=lr)
        loss = nn.train(X_train, y_train, epochs=200, method='batch')
        plt.plot(loss, label=f"Alfa (learning rate) = {lr}")
    plt.title("Vliv konstanty učení na konvergenci")
    plt.xlabel("Epocha")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    for name, topo in topologies.items():
        nn = SimpleNeuralNetwork(layer_sizes=topo, learning_rate=0.05)
        loss = nn.train(X_train, y_train, epochs=200, method='batch')
        plot_boundaries(nn, X_all, y_all, f"NN {name}")

