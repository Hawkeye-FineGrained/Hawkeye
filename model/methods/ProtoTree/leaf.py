import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import Node


class Leaf(Node):

    def __init__(self,
                 index: int,
                 num_classes: int,
                 args
                 ):
        super().__init__(index)

        # Initialize the distribution parameters
        if args.disable_derivative_free_leaf_optim:
            self._dist_params = nn.Parameter(torch.randn(num_classes), requires_grad=True)
        elif args.kontschieder_normalization:
            self._dist_params = nn.Parameter(torch.ones(num_classes), requires_grad=False)
        else:
            self._dist_params = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = args.log_probabilities

        self._kontschieder_normalization = args.kontschieder_normalization

    def forward(self, xs: torch.Tensor, **kwargs):

        # Get the batch size
        batch_size = xs.size(0)

        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault('attr', dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        if not self._log_probabilities:
            node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
        else:
            node_attr.setdefault((self, 'pa'), torch.zeros(batch_size, device=xs.device))

        # Obtain the leaf distribution
        dist = self.distribution()  # shape: (k,)
        # Reshape the distribution to a matrix with one single row
        dist = dist.view(1, -1)  # shape: (1, k)
        # Duplicate the row for all x in xs
        dists = torch.cat((dist,) * batch_size, dim=0)  # shape: (bs, k)

        # Store leaf distributions as node property
        node_attr[self, 'ds'] = dists

        # Return both the result of the forward pass as well as the node properties
        return dists, node_attr

    def distribution(self) -> torch.Tensor:
        if not self._kontschieder_normalization:
            if self._log_probabilities:
                return F.log_softmax(self._dist_params, dim=0)
            else:
                # Return numerically stable softmax (see http://www.deeplearningbook.org/contents/numerical.html)
                return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0)

        else:
            # kontschieder_normalization's version that uses a normalization factor instead of softmax:
            if self._log_probabilities:
                return torch.log((self._dist_params / torch.sum(
                    self._dist_params)) + 1e-10)  # add small epsilon for numerical stability
            else:
                return self._dist_params / torch.sum(self._dist_params)

    @property
    def requires_grad(self) -> bool:
        return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    @property
    def size(self) -> int:
        return 1

    @property
    def leaves(self) -> set:
        return {self}

    @property
    def branches(self) -> set:
        return set()

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self}

    @property
    def num_branches(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 0
