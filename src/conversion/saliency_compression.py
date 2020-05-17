def find_minimum_index(saliency_matrix: np.ndarray, should_ignore: Set[int]) -> Tuple[int, int]:
    smallest_saliency = BIG_NUMBER
    best_index = None

    for i in range(saliency_matrix.shape[0]):
        if i in should_ignore:
            continue
    
        for j in range(saliency_matrix.shape[1]):
            if j in should_ignore or i == j:  # Avoid diagonal entries
                continue

            if saliency_matrix[i, j] < smallest_saliency:
                smallest_saliency = saliency_matrix[i, j]
                best_index = (i, j)

    return best_index


def compress_layer(weights: np.ndarray, next_layer: np.ndarray, remove_frac: float) -> np.ndarray:
    """
    Compresses the layer's weights using data free parameter pruning
    """
    assert remove_frac >= 0 and remove_frac < 1.0, 'Remove fraction must be in the range [0, 1)'

    if abs(remove_frac - 1.0) < SMALL_NUMBER:
        return weights

    # 1) Compute the saliency matrix
    N = weights.shape[1]
    weight_diffs = np.zeros((N, N))
    activations = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            diff = np.square(np.linalg.norm(weights[:, i] - weights[:, j], ord=2))
            avg_nxt = np.average(np.square(next_layer[j, :]))

            weight_diffs[i, j] = diff
            activations[i, j] = avg_nxt

    num_to_remove = int(remove_frac * N)
    removed_neurons: Set[int] = set()
    
    pruned_next_layer = np.copy(next_layer)

    for _ in range(num_to_remove):
        # 2) Find the minimum entry
        i, j = find_minimum_index(weight_diffs * activations, should_ignore=removed_neurons)

        # 3) Prune the weights
        activations[:, i] += activations[:, j]  # Merge the activations
        pruned_next_layer[i, :] += pruned_next_layer[j, :]
        removed_neurons.add(j)

    kept_neurons = list(sorted(set(range(N)).difference(removed_neurons)))
    compressed_weights = weights[:, kept_neurons]
    compressed_next_layer = pruned_next_layer[kept_neurons, :]

    return compressed_weights, compressed_next_layer


