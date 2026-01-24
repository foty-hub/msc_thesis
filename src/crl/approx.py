import faiss
import numpy as np


class FaissKDTree:
    def __init__(self, X, leaf_size=40, metric="minkowski", **kwargs):
        self.data = np.ascontiguousarray(X.astype("float32"))
        self.n_samples, self.d = self.data.shape
        self.index = faiss.IndexFlatL2(self.d)
        self.index.add(self.data)

    def query(
        self,
        X,
        k=1,
        return_distance=True,
        dualtree=False,
        breadth_first=False,
        sort_results=True,
    ):
        X = np.ascontiguousarray(X.astype("float32"))

        # Auto-reshape 1D vectors
        if X.ndim == 1:
            X = X.reshape(1, -1)

        distances_sq, indices = self.index.search(X, k)

        if return_distance:
            return np.sqrt(distances_sq), indices
        return indices

    def query_radius(self, X, r, count_only=False, sort_results=False):
        X = np.ascontiguousarray(X.astype("float32"))

        if X.ndim == 1:
            X = X.reshape(1, -1)

        lims, D_sq, I = self.index.range_search(X, r**2)

        if count_only:
            return lims[1:] - lims[:-1]

        n_queries = X.shape[0]
        indices = np.empty(n_queries, dtype=object)
        distances = np.empty(n_queries, dtype=object)

        for i in range(n_queries):
            start, end = lims[i], lims[i + 1]
            indices[i] = I[start:end]
            distances[i] = np.sqrt(D_sq[start:end])

            if sort_results and len(indices[i]) > 0:
                sort_idx = np.argsort(distances[i])
                indices[i] = indices[i][sort_idx]
                distances[i] = distances[i][sort_idx]

        return (distances, indices)

    def get_arrays(self):
        return self.data, np.arange(self.n_samples), None, None


def main():
    import numpy as np

    x = np.random.uniform(size=(100_000, 10))
    tree = FaissKDTree(x)
    dists, inds = tree.query(x[0], k=5)
    print(inds)


if __name__ == "__main__":
    main()
