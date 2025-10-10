from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.space import Space


class GraphInstance(NamedTuple):
    nodes: np.ndarray
    edges: Optional[np.ndarray]
    edge_links: Optional[np.ndarray]


class GraphSpace(Space):
    """A minimal Graph space: nodes x node_space, edges x edge_space, plus edge_links.

    This is a lightweight adaptation of the Gym Graph space used for variable-size
    graph observations. It is not flattenable.
    """

    def __init__(self, node_space: Union[Box, Discrete], edge_space: Optional[Union[Box, Discrete]] = None):
        assert isinstance(node_space, (Box, Discrete)), "node_space must be Box or Discrete"
        if edge_space is not None:
            assert isinstance(edge_space, (Box, Discrete)), "edge_space must be Box or Discrete or None"
        self.node_space = node_space
        self.edge_space = edge_space
        super().__init__(None, None)

    @property
    def is_np_flattenable(self):
        return False

    def sample(self, num_nodes: int = 1, num_edges: Optional[int] = None) -> GraphInstance:
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        if num_edges is None:
            num_edges = max(0, num_nodes - 1)
        # sample node features
        node_shape = (num_nodes,) + self.node_space.shape
        nodes = np.random.uniform(low=self.node_space.low, high=self.node_space.high, size=node_shape).astype(self.node_space.dtype)
        if self.edge_space is None or num_edges == 0:
            return GraphInstance(nodes, None, None)
        edge_shape = (num_edges,) + self.edge_space.shape
        edges = np.random.uniform(low=self.edge_space.low, high=self.edge_space.high, size=edge_shape).astype(self.edge_space.dtype)
        edge_links = np.random.randint(low=0, high=num_nodes, size=(num_edges, 2), dtype=np.int32)
        return GraphInstance(nodes, edges, edge_links)

    def contains(self, x: GraphInstance) -> bool:
        if not isinstance(x, GraphInstance):
            return False
        if not isinstance(x.nodes, np.ndarray):
            return False
        if x.edges is None and x.edge_links is None:
            return True
        if x.edges is None or x.edge_links is None:
            return False
        if x.edges.shape[0] != x.edge_links.shape[0]:
            return False
        return True

    def __repr__(self) -> str:
        return f"GraphSpace(node_space={self.node_space}, edge_space={self.edge_space})"
