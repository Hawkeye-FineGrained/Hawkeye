import torch.nn as nn


class Node(nn.Module):

    def __init__(self, index: int):
        super().__init__()
        self._index = index

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def index(self) -> int:
        return self._index

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def nodes(self) -> set:
        return self.branches.union(self.leaves)

    @property
    def leaves(self) -> set:
        raise NotImplementedError

    @property
    def branches(self) -> set:
        raise NotImplementedError

    @property
    def nodes_by_index(self) -> dict:
        raise NotImplementedError

    @property
    def num_branches(self) -> int:
        return len(self.branches)

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    @property
    def depth(self) -> int:
        raise NotImplementedError
