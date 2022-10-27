import torch
import numpy as np

from .node import Node


class Branch(Node):

    def __init__(self,
                 index: int,
                 l: Node,
                 r: Node,
                 args
                 ):
        super().__init__(index)
        self.l = l
        self.r = r

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = args.log_probabilities

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
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
        else:
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))

        # Obtain the probabilities of taking the right subtree
        ps = self.g(xs, **kwargs)  # shape: (bs,)

        if not self._log_probabilities:
            # Store decision node probabilities as node attribute
            node_attr[self, 'ps'] = ps
            # Store path probabilities of arriving at child nodes as node attributes
            node_attr[self.l, 'pa'] = (1 - ps) * pa
            node_attr[self.r, 'pa'] = ps * pa
            # # Store alpha value for this batch for this decision node
            # node_attr[self, 'alpha'] = torch.sum(pa * ps) / torch.sum(pa)

            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.l.forward(xs, **kwargs)  # shape: (bs, k)
            r_dists, _ = self.r.forward(xs, **kwargs)  # shape: (bs, k)
            # Weight the probability distributions by the decision node's output
            ps = ps.view(batch_size, 1)
            return (1 - ps) * l_dists + ps * r_dists, node_attr  # shape: (bs, k)
        else:
            # Store decision node probabilities as node attribute
            node_attr[self, 'ps'] = ps

            # Store path probabilities of arriving at child nodes as node attributes
            # source: rewritten to pytorch from
            # https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/math/generic.py#L447-L471
            x = torch.abs(ps) + 1e-7  # add small epsilon for numerical stability
            oneminusp = torch.where(x < np.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

            node_attr[self.l, 'pa'] = oneminusp + pa
            node_attr[self.r, 'pa'] = ps + pa

            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.l.forward(xs, **kwargs)  # shape: (bs, k)
            r_dists, _ = self.r.forward(xs, **kwargs)  # shape: (bs, k)

            # Weight the probability distributions by the decision node's output
            ps = ps.view(batch_size, 1)
            oneminusp = oneminusp.view(batch_size, 1)
            logs_stacked = torch.stack((oneminusp + l_dists, ps + r_dists))
            return torch.logsumexp(logs_stacked, dim=0), node_attr  # shape: (bs,)

    def g(self, xs: torch.Tensor, **kwargs):
        out_map = kwargs['out_map']  # Obtain the mapping from decision nodes to conv net outputs
        conv_net_output = kwargs['conv_net_output']  # Obtain the conv net outputs
        out = conv_net_output[out_map[self]]  # Obtain the output corresponding to this decision node
        return out.squeeze(dim=1)

    @property
    def size(self) -> int:
        return 1 + self.l.size + self.r.size

    @property
    def leaves(self) -> set:
        return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) -> set:
        return {self} \
            .union(self.l.branches) \
            .union(self.r.branches)

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self,
                **self.l.nodes_by_index,
                **self.r.nodes_by_index}

    @property
    def num_branches(self) -> int:
        return 1 + self.l.num_branches + self.r.num_branches

    @property
    def num_leaves(self) -> int:
        return self.l.num_leaves + self.r.num_leaves

    @property
    def depth(self) -> int:
        return self.l.depth + 1
