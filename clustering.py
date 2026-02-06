from generic import load_data, plot_data, np

def clustering_level(data):
    """
    Metoda shlukové hladiny
    - sloučí postupně všechny body do 1 clusteru
    - vezme vzdálenosti slučování a podívá se, kdy došlo k největší změně
    - vrátí počet shluků v datech
    """
    n = len(data)
    if n == 0: return 0
    distances = []

    # 1. Předvýpočet matice vzdáleností (všechny body proti všem)
    # Využíváme "broadcasting" pro extrémně rychlý výpočet
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Pro diagonálu (vzdálenost bodu se sebou samým) nastavíme nekonečno
    np.fill_diagonal(dist_matrix, np.inf)

    while True:
        # 2. Najdeme globální minimum v matici vzdáleností
        min_dist = np.min(dist_matrix)
        if min_dist == np.inf:
            break

        # Získáme indexy dvou nejbližších shluků
        # argmin nám vrátí lineární index, rozbalíme ho na (row, col)
        idx1, idx2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # 3. Aktualizace matice
        # shluk idx2 "vyloučíme" tím, že mu nastavíme nekonečné vzdálenosti
        # shluk idx1 aktualizujeme (např. Single Linkage = minimum z obou)
        dist_matrix[idx1, :] = np.minimum(dist_matrix[idx1, :], dist_matrix[idx2, :])
        dist_matrix[:, idx1] = dist_matrix[idx1, :] # symetrie

        dist_matrix[idx2, :] = np.inf
        dist_matrix[:, idx2] = np.inf
        np.fill_diagonal(dist_matrix, np.inf) # reset diagonály

        # Uložíme vzdálenost bodů, které jsme sloučili pro pozdější hledání hladiny
        distances.append(min_dist)

    # Získáme vzdálenosti slučování z metody shlukových hladin:
    dists_np = np.array(distances, dtype=np.float32)
    # Spočítáme si skoky mezi jednotlivými hladinami:
    dist_jumps = dists_np[1:] - dists_np[:-1]
    # 4. Zjistíme, ve kterém kroku došlo k největšímu skoku (naše shlukovací hladina) a vyvodíme počet shluků:
    n = len(distances) - np.argmax(dist_jumps).astype(int)
    # (Není to zcela nejspolehlivější způsob, například, kdyby poslední shluk byl od předchozích výrazně dále,
    # třeba 2x, získali bychom pouze 2 shluky, protože by došlo k ještě většímu hladinovému skoku,
    # ale v tomto příkladě, s těmito daty, je tato metoda dostačující)

    return n


def chain_map(data, start_node=0):
    """
    Metoda řetězové mapy
    - spojí všechny bode podle nejkratších vzdáleností jedním "řetězem"
    - vezme vzdálenosti "článků řetězu" a podívá se, kdy došlo k největším skokům
    - vrátí počet shluků v datech
    """
    n = len(data)
    if n == 0: return 0

    # Pole pro ukládání nejmenší vzdálenosti každého bodu k aktuálnímu řetězci
    # Na začátku nastavíme nekonečno
    min_dists_to_chain = np.full(n, np.inf)

    # Seznam, kam uložíme finální délky hran řetězce
    chain_distances = []

    # Maska navštívených bodů
    visited = np.zeros(n, dtype=bool)

    # Začneme u prvního bodu
    current_node = start_node
    visited[current_node] = True

    for _ in range(n - 1):
        # 1. Vektorizovaný výpočet vzdáleností od posledního přidaného bodu
        # ke všem ostatním (nenavštíveným)
        diff = data - data[current_node]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))

        # 2. Aktualizujeme nejmenší známé vzdálenosti k řetězci
        # (Bod si pamatuje buď svou starou nejmenší vzdálenost, nebo novou k právě přidanému bodu)
        min_dists_to_chain = np.minimum(min_dists_to_chain, dists)

        # 3. Vybereme nenavštívený bod, který je nejblíže k řetězci
        # Maskujeme již navštívené body (nastavíme jim dočasně nekonečno)
        temp_dists = np.where(visited, np.inf, min_dists_to_chain)
        next_node = np.argmin(temp_dists)

        # 4. Uložíme vzdálenost, na které jsme bod připojili
        chain_distances.append(temp_dists[next_node])

        # Přesuneme se na nový bod
        visited[next_node] = True
        current_node = next_node

    chain_distances = np.array(chain_distances)
    # 5. Najdeme hraniční délku skoku:
    h = (np.average(chain_distances)+np.amax(chain_distances))/2
    # (Opět to není nejlepší způsob, naráží na stejný problém, jako předchozí metoda,
    # ale v tomto příkladě, s těmito daty, je tento přístup dostačující)
    # VÝSLEDEK: Spočítáme, kolik hran v řetězci je delších než H
    breaks = np.sum(chain_distances > h)

    return breaks + 1


def maximin(data, q=0.7, start_node=0):
    """
    Metoda MAXIMIN
    - vezme vždy bod nejvzdálenejší od všech již nalezených "středů"
    - podívá se, zda je splněna podmínka se vzdáleností od nejbližšího středu a označí bod jako další střed
    - pokud podmínka splněna nebyla, končí a vrátí počet shluků v datech
    """
    n = len(data)
    if n == 0: return 0

    # Seznam indexů nalezených středů
    center_indices = [start_node]

    # Udržujeme si pole aktuálních minimálních vzdáleností ke středům
    # Na začátku: vzdálenosti všech bodů k prvnímu středu
    diff = data - data[start_node]
    min_distances = np.sqrt(np.sum(diff ** 2, axis=1))

    while True:
        # 1. Najdeme bod, který je nejdále od svého nejbližšího středu
        new_center_idx = np.argmax(min_distances)
        max_min_val = min_distances[new_center_idx]

        # 2. Získáme průměr vzdáleností středů (pro pozdější testování)
        diff = data[center_indices, np.newaxis, :] - data[np.newaxis, center_indices, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        mask = ~np.eye(dist_matrix.shape[0], dtype=bool)
        avg_dist = dist_matrix[mask if mask.any() else 0].mean()
        # V prvmín běhu cheme nechat podmínu vždy proběhnout, proto "else 0", získáme tak avg_dist = 0

        # 3. Pokud je tato vzdálenost větší než q * průměr vzdáleností středů, máme nový střed
        if max_min_val > q*avg_dist:
            center_indices.append(new_center_idx)

            # Vypočteme vzdálenosti od nového středu ke všem bodům
            new_diff = data - data[new_center_idx]
            new_dists = np.sqrt(np.sum(new_diff ** 2, axis=1))

            # Bod si ponechá buď starou minimální vzdálenost, nebo novou (k novému středu)
            min_distances = np.minimum(min_distances, new_dists)
        else:
            break

    # Vzhledem k povaze a rozmístění dat v prostoru je výsledek této metody VELMI citlivý jak na volbu parametru q,
    # tak na počátečím bodě (volbě prvního středu)
    return len(center_indices)


def standard_kmeans(data, k, max_iters=100):
    # 1. Náhodně vybereme k bodů jako počáteční středy (centroidy)
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices.astype(int)]
    labels = np.array([])

    for _ in range(max_iters):
        # 2. Přiřadíme body k nejbližšímu centroidu
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 3. Přepočítáme centroidy (těžiště bodů v daném shluku)
        new_centroids = np.array([
            data[labels == i].mean(axis=0) if len(data[labels == i]) > 0 else centroids[i]
            for i in range(k)
        ])

        # 4. Pokud se centroidy už nemění, skončíme dřív
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


def calculate_sse(data, centroids, labels):
    """Vypočítá Sumu čtverců chyb (SSE) pro vyhodnocení kvality."""
    sse = 0
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - centroids[i]) ** 2)
    return sse


def direct_partitioning(data, k):
    centroids, labels = standard_kmeans(data, k)
    sse = calculate_sse(data, centroids, labels)
    return centroids, labels, sse


def bisecting_kmeans(data, k):
    # Na začátku máme jeden seznam obsahující indexy všech bodů
    clusters_indices = [np.arange(len(data))]  # [[0, 1, 2, 3, 4, ...]]
    centroids = [data.mean(axis=0)]  # střed všech bodů, první centroid

    while len(clusters_indices) < k:
        # 1. Najdeme shluk s největším SSE
        sses = []
        for i, idx_list in enumerate(clusters_indices):
            cluster_points = data[idx_list]
            sse = np.sum((cluster_points - centroids[i]) ** 2)
            sses.append(sse)

        # Shluk, který budeme dělit
        split_cluster_idx = np.argmax(sses)
        indices_to_split = clusters_indices.pop(split_cluster_idx)
        points_to_split = data[indices_to_split]
        # Odstraníme příslušný centroid, bude nahrazen 2 novými
        centroids.pop(split_cluster_idx)

        # 2. Rozdělíme na dva pomocí 2-means
        # sub_labels jsou 0 a 1 v rámci tohoto podvýběru
        cs, sub_labels = standard_kmeans(points_to_split, k=2)
        centroids.extend(cs)

        # 3. Rozdělíme původní indexy podle sub_labels a vrátíme do seznamu
        clusters_indices.append(indices_to_split[sub_labels == 0])
        clusters_indices.append(indices_to_split[sub_labels == 1])

    # --- FINÁLNÍ SKLÁDÁNÍ VÝSLEDKŮ ---
    final_labels = np.zeros(len(data), dtype=int)
    final_centroids = np.array(centroids, dtype=np.float32)
    for cluster_id, idx_list in enumerate(clusters_indices):
        # Přiřadíme ID shluku bodům na správných indexech
        final_labels[idx_list] = cluster_id
    total_sse = calculate_sse(data, final_centroids, final_labels)
    return final_centroids, final_labels, total_sse


def iterative_optimization(data, initial_labels, R):
    n, d = data.shape
    labels = initial_labels.copy().astype(int)

    # 1. Inicializace s a střední hodnoty
    s = np.bincount(labels, minlength=R).astype(float)
    centroids = np.array([data[labels == r].mean(axis=0) if s[r] > 0
                          else np.zeros(d) for r in range(R)])

    iteration = 0
    while True:
        iteration += 1
        any_moved = False

        # 2. Procházíme body
        for k in range(n):
            i = labels[k]
            x_hat = data[k]

            # 3. Podmínka s_i > 1
            if s[i] <= 1:
                continue

            # Spočítáme čtverce vzdáleností od x_hat ke všem těžištím najednou
            # (R, d) - (d,) -> (R, d) -> sum(axis=1) -> (R,)
            dists_sq = np.sum((centroids - x_hat) ** 2, axis=1)

            # Připravíme si pole koeficientů (R,)
            coeffs = s / (s + 1)
            A_all = coeffs * dists_sq
            A_all[i] = np.inf  # abychom při hledání min A_j nevybrali aktuální shluk i

            # Hodnota pro aktuální shluk i
            A_i = (s[i] / (s[i] - 1)) * dists_sq[i]

            # Najdeme j* (index nejmenšího přírůstku)
            j_star = np.argmin(A_all)
            A_j_star = A_all[j_star]

            # 4. Rozhodneme o přesunu
            if A_i > A_j_star:
                print("Moving a point")
                # 5. Aktualizace středy a s
                centroids[i] -= (x_hat - centroids[i]) / (s[i] - 1)
                centroids[j_star] += (x_hat - centroids[j_star]) / (s[j_star] + 1)
                s[i] -= 1
                s[j_star] += 1
                labels[k] = j_star
                any_moved = True

        if not any_moved:
            break

    return centroids, labels


if __name__ == '__main__':
    # cl_cm_mm = 0b111    # Jednoduchý přepínač, které metody shlukování chceme spustit
    cl_cm_mm = 0b000
    data_list = load_data("data_shl.txt")
    data_p = [(p[0], p[1]) for p in data_list]
    data = np.array(data_p, dtype=np.float32)

    # Shluková hladina
    if cl_cm_mm & 0b100:
        print("Number of clusters (by clustering levels):", clustering_level(data))

    # Řetězová mapa
    if cl_cm_mm & 0b010:
        for n in np.random.randint(0, len(data), 10):
            print(f"Number of clusters (by chain map, start_node={n:4}):", chain_map(data, n))

    # MAXIMIN
    if cl_cm_mm & 0b001:
        for q in np.arange(0.4, 0.6, 0.02, dtype=np.float32):
            n = np.random.randint(0, len(data), 1)[0]
            print(f"Number of clusters (by maximin, {q=:.2f}, {n=:4}):", maximin(data, q, n))

    # ----------------
    # K-means
    K_TARGET = 3  # Počet shluků, který nám vyšel z předchozích metod (MAXIMIN atd.)

    # Přímé dělení
    c_dir, l_dir, sse_dir = direct_partitioning(data, K_TARGET)
    print(f"Přímé dělení - SSE: {sse_dir:.2f}")
    plot_data(data_p, l_dir)

    # Nerovnoměrné binární dělení
    c_bis, l_bis, sse_bis = bisecting_kmeans(data, K_TARGET)
    print(f"Binární dělení - SSE: {sse_bis:.2f}")
    plot_data(data_p, l_bis)

    opt_centroids, opt_labels = iterative_optimization(data, l_bis, 3)
    print(f"Iterativní optimalizace: {"Optimalizováno" if (opt_labels-l_bis).any() else "Beze změny"}")

