import numpy as np
from typing import Sequence, Callable
from crl.cons.buffer import Transition


# Adaptive Tree Tiling
class _KDNode:
    __slots__ = ("axis", "thresh", "left", "right", "leaf_id", "is_leaf")

    def __init__(
        self,
        *,
        axis: int | None = None,
        thresh: float | None = None,
        left: "_KDNode | None" = None,
        right: "_KDNode | None" = None,
        leaf_id: int | None = None,
        is_leaf: bool = False,
    ):
        self.axis = axis
        self.thresh = thresh
        self.left = left
        self.right = right
        self.leaf_id = leaf_id
        self.is_leaf = is_leaf


def build_kdtree_tiling(
    buffer: Sequence[Transition],
    min_leaf: int = 32,
    include_actions: bool = False,
) -> tuple[list[int], Callable[[np.ndarray, np.ndarray | float | None], int], int]:
    """
    Build a variance-split kd-tree over 4-D states (or 5-D state+action).

    Returns
    -------
    ids                 : leaf id for each transition in `buffer`
    discretise(obs, act): maps observation (+action) to a leaf id
    n_discrete_states   : total number of leaves
    """
    # ------------------------------------------------------------------ data
    if include_actions:
        states = np.stack(
            [np.concatenate([tr.state[0], tr.action]) for tr in buffer],
            axis=0,
        )
        expected_dim = 5
    else:
        states = np.stack([tr.state[0] for tr in buffer], axis=0)
        expected_dim = 4

    N, D = states.shape
    if D != expected_dim:
        raise ValueError(f"Expected {expected_dim}-D inputs but got {D}-D.")

    # ------------------------------------------------------------ tree build
    next_leaf_id = [0]

    def _build(idx: np.ndarray) -> _KDNode:
        if idx.size <= min_leaf:
            leaf = _KDNode(leaf_id=next_leaf_id[0], is_leaf=True)
            next_leaf_id[0] += 1
            return leaf

        subset = states[idx]
        for axis in subset.var(axis=0).argsort()[::-1]:  # try axes by variance
            thresh = float(np.median(subset[:, axis]))
            left_mask = subset[:, axis] <= thresh
            right_mask = ~left_mask
            if left_mask.any() and right_mask.any():  # valid split
                node = _KDNode(axis=int(axis), thresh=thresh)
                node.left = _build(idx[left_mask])
                node.right = _build(idx[right_mask])
                return node

        # no axis gives a valid split → make leaf
        leaf = _KDNode(leaf_id=next_leaf_id[0], is_leaf=True)
        next_leaf_id[0] += 1
        return leaf

    root = _build(np.arange(N))

    # -------------------------------------------------------------- lookup
    def _lookup(node: _KDNode, x: np.ndarray) -> int:
        while not node.is_leaf:
            node = node.left if x[node.axis] <= node.thresh else node.right
        return node.leaf_id

    # ----------------------------------------------------------- interface
    def discretise(
        obs: np.ndarray,
        action: np.ndarray | float | None = None,
    ) -> int | np.ndarray:
        """
        * include_actions=False: discretise(obs) where obs shape (4,) or (batch,4)
        * include_actions=True : discretise(obs, action)
              - obs shape (4,) & action scalar → int
              - obs shape (batch,4) & action scalar or (batch,) array → array
        """
        arr = np.asarray(obs, dtype=float)
        if action:
            action = action[0]

        if include_actions:
            if action is None:
                raise ValueError("Action must be provided when include_actions=True.")

            act_arr = np.asarray(action, dtype=float)
            if arr.ndim == 1:
                if arr.shape[0] != 4 or act_arr.ndim > 0:
                    raise ValueError("Expect obs (4,) and action scalar.")
                x = np.concatenate([arr, [act_arr.item()]])
                return _lookup(root, x)

            elif arr.ndim == 2 and arr.shape[1] == 4:
                if act_arr.ndim == 0:  # scalar → broadcast
                    act_arr = np.full(arr.shape[0], act_arr.item())
                if act_arr.shape != (arr.shape[0],):
                    raise ValueError("Action array must match batch length.")
                batch = np.hstack([arr, act_arr[:, None]])
                return np.fromiter(
                    (_lookup(root, row) for row in batch), int, count=batch.shape[0]
                )

            else:
                raise ValueError(
                    "Obs must be (4,) or (batch,4) when include_actions=True."
                )

        else:  # no actions
            if arr.ndim == 1:
                if arr.shape[0] != 4:
                    raise ValueError("Expected obs shape (4,).")
                return _lookup(root, arr)
            elif arr.ndim == 2 and arr.shape[1] == 4:
                return np.fromiter(
                    (_lookup(root, row) for row in arr), int, count=arr.shape[0]
                )
            else:
                raise ValueError("Obs must be (4,) or (batch,4).")

    # -------------------------------------------------------- existing ids
    ids = [_lookup(root, s) for s in states]
    n_discrete_states = max(ids) + 1
    print(f"Built tree with {n_discrete_states:,} nodes")
    return ids, discretise, n_discrete_states
