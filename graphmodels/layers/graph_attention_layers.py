"""Graph attention layers"""

import torch
import torch.nn.functional as F  # noqa: N812
import torch_scatter
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker


@jt(typechecker=typechecker)
class GraphAttentionLayerSkip(nn.Module):
    """Implements a graph attention layer with skip connection.

    After the aggregation step, we add the transformed node features
    to the aggregated node features.

    Attributes
        n_node_features: number of input node features.
        n_hidden_features: number of hidden features in intermediate layers.
        scaling: scaling constant for LeakyRelu
    """

    def __init__(
        self,
        n_node_features: int,
        n_hidden_features: int,
        scaling: float = 0.2,
        dropout: float = 0.25,
        apply_act: bool = False,
    ):
        super().__init__()
        self.scaling = scaling
        self.apply_act = apply_act
        self.dropout = nn.Dropout(p=dropout)
        self.w = nn.Linear(n_node_features, n_hidden_features)
        self.attn = nn.Linear(n_hidden_features * 2, 1, bias=False)

    def compute_attention(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "attention_score 1"],  # noqa: F722
        Float[torch.Tensor, "nodes hidden_features"],  # noqa: F722
        Int[torch.Tensor, " target_index"],
    ]:
        """Computes attention score between nodes i and j.

        The attention mechanism is defined as:

            α_ij=softmax_j( LeakyReLU( aᵀ [ W h_i || W h_j ] ) )

        where:
        - h_i, h_j are the input features of nodes i and j,
        - W is a shared learnable weight matrix,
        - a is a learnable attention vector,
        - || denotes vector concatenation,
        - softmax_j is applied over all neighbors j ∈ N(i) of node i.

        The updated feature for node i is computed as:

            h_i'=σ( Σ_{j ∈ N(i)} α_ij · W h_j )

        where σ is a non-linear activation function.

        Args:
            node_features: input node features (shape=N, F)
            where N is the number of nodes and F the number of features.
            edge_index: graph connectivity in COO format with shape (2, E),
                where E is the number of edges. The first row contains target
                node indices, and the second row contains source node indices.
        Returns:
            Attention scores for nodes.
        """
        node_features = self.dropout(node_features)
        h = self.w(node_features)

        neighbors_nodes = edge_index[1]
        target_nodes = edge_index[0]

        h_i = h[target_nodes]

        h_j = h[neighbors_nodes]
        h_concat = torch.cat([h_i, h_j], dim=-1)

        eij = F.leaky_relu(self.attn(h_concat), negative_slope=self.scaling)

        attention_score = self.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )
        message = attention_score * h_j

        return message, h, target_nodes

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> Float[torch.Tensor, "nodes node_features"]:
        """Performs the forward pass.

        Args:
            node_features: input features for the nodes.
            edge_index: a matrix indicating which nodes are connected.
            COO format (i.e. 2, N_EDGES).
        Returns:
            Updated node features after attention.

        """
        message, transformed_node_features, target_nodes = (
            self.compute_attention(
                node_features=node_features,
                edge_index=edge_index,
            )
        )
        out = torch_scatter.scatter_add(
            message,
            target_nodes,
            dim=0,
            dim_size=transformed_node_features.size(0),
        )
        out = out + transformed_node_features
        if self.apply_act:
            out = F.elu(out)
        return self.dropout(out)


@jt(typechecker=typechecker)
class GraphAttentionLayerEdge(nn.Module):
    """Implements a graph attention layer with edge features.

    Attributes
        n_node_features: number of input node features.
        n_hidden_features: number of hidden features in intermediate layers.
        scaling: scaling constant for LeakyRelu
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden_features: int,
        scaling: float = 0.2,
        dropout: float = 0.25,
        apply_act: bool = False,
    ):
        super().__init__()
        self.scaling = scaling
        self.apply_act = apply_act
        self.dropout = nn.Dropout(dropout)
        self.w = nn.Linear(n_node_features, n_hidden_features)
        self.edgew = nn.Linear(n_edge_features, n_hidden_features)
        self.attn = nn.Linear(n_hidden_features * 3, 1, bias=False)

    def compute_attention(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "attention_score 1"],  # noqa: F722
        Float[torch.Tensor, "nodes hidden_features"],  # noqa: F722
        Float[torch.Tensor, "edges hidden_features"],  # noqa: F722
        Int[torch.Tensor, " target_index"],
    ]:
        """Computes attention score between nodes i and j.

        The attention mechanism is defined as:

            α_ij=softmax_j( LeakyReLU( aᵀ [ W h_i || W h_j ] ) )

        where:
        - h_i, h_j are the input features of nodes i and j,
        - W is a shared learnable weight matrix,
        - a is a learnable attention vector,
        - || denotes vector concatenation,
        - softmax_j is applied over all neighbors j ∈ N(i) of node i.

        The updated feature for node i is computed as:

            h_i'=σ( Σ_{j ∈ N(i)} α_ij · W h_j )

        where σ is a non-linear activation function.

        Args:
            node_features: Input node features.
            edge_features: Input edge features.
            edge_index: Graph connectivity in COO format with shape $(2, E)$,
            where $E$ is the number of edges.

        Returns:
            Attention scores multiplied by transformed neighbor features,
            transformed node features, and target node indices.
        """

        edge_features = self.dropout(edge_features)
        node_features = self.dropout(node_features)
        h = self.w(node_features)
        edge_h = self.edgew(edge_features)

        neighbors_nodes = edge_index[1]
        target_nodes = edge_index[0]

        h_i = h[target_nodes]
        h_j = h[neighbors_nodes]
        h_concat = torch.cat([h_i, h_j, edge_h], dim=-1)

        eij = F.leaky_relu(self.attn(h_concat), negative_slope=self.scaling)

        attention_score = F.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )
        message = attention_score * h_j

        return message, h, edge_h, target_nodes

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "nodes node_features"],
        Float[torch.Tensor, "edges edge_features"],
    ]:
        """Performs the forward pass.

        Args:
            node_features: input features for the nodes.
            edge_features: input features for the edges.
            edge_index: a matrix indicating which nodes are connected.
            COO format (i.e. 2, N_EDGES).
        Returns:
            Updated node features after attention step.
            The update includes edge features.

        """
        message, transformed_node_features, edge_h, target_nodes = (
            self.compute_attention(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
            )
        )

        out = torch_scatter.scatter_add(
            message,
            target_nodes,
            dim=0,
            dim_size=node_features.size(0),
        )
        out = out + transformed_node_features
        if self.apply_act:
            out = F.elu(out)
            edge_h = F.elu(edge_h)
        return self.dropout(out), self.dropout(edge_h)


@jt(typechecker=typechecker)
class MultiHeadGATLayer(nn.Module):
    """
    Implements a multihead graph attention layer with skip connections.

    This layer aims to be similar to the torch geometric implementation
    by using reshaping operations.

    You can:
    - Apply non-linearity to the final output using `apply_act`.
    - Concatenate or average the outputs of the attention step using `concat`.
    - Add skip connections using `add_skip_connection`.

    Attributes:
        n_node_features: Number of input node features.
        n_hidden_features: Number of hidden features.
        scaling: Negative slope for LeakyReLU.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        apply_act: Whether to apply non-linearity to the final output.
        concat: If True, concatenate the outputs of all heads.
        Else, average them.
        add_skip_connection: Whether to add skip connections.
    """

    def __init__(
        self,
        n_node_features: int,
        n_hidden_features: int,
        num_heads: int = 1,
        scaling: float = 0.2,
        dropout: float = 0.0,
        apply_act: bool = False,
        concat: bool = True,
        add_skip_connection: bool = True,
    ):
        super().__init__()
        self.scaling = scaling
        self.concat = concat
        self.apply_act = apply_act
        self.num_heads = num_heads
        self.n_hidden_features = n_hidden_features
        self.n_node_features = n_node_features
        self.dropout = nn.Dropout(p=dropout)
        self.add_skip_connection = add_skip_connection

        self.w = nn.Linear(n_node_features, n_hidden_features * num_heads)
        self.attn = nn.Parameter(
            torch.Tensor(1, num_heads, 2 * n_hidden_features),
        )
        nn.init.xavier_uniform_(self.attn.data)

    def compute_attention(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "attention_score 1"],  # noqa: F722
        Float[torch.Tensor, "nodes hidden_features"],  # noqa: F722
        Int[torch.Tensor, " target_index"],
    ]:
        """Computes attention score between nodes i and j.

        The attention mechanism is defined as:

            α_ij=softmax_j( LeakyReLU( aᵀ [ W h_i || W h_j ] ) )

        where:
        - h_i, h_j are the input features of nodes i and j,
        - W is a shared learnable weight matrix,
        - a is a learnable attention vector,
        - || denotes vector concatenation,
        - softmax_j is applied over all neighbors j ∈ N(i) of node i.

        The updated feature for node i is computed as:

            h_i'=σ( Σ_{j ∈ N(i)} α_ij · W h_j )

        where σ is a non-linear activation function.

        Args:
            node_features: input node features (shape=N, F)
            where N is the number of nodes and F the number of features.
            edge_index: graph connectivity in COO format with shape (2, E),
                where E is the number of edges. The first row contains target
                node indices, and the second row contains source node indices.
        Returns:
            Attention scores for nodes.
        """
        h = self.dropout(self.w(node_features)).view(
            -1,
            self.num_heads,
            self.n_hidden_features,
        )

        neighbors_nodes = edge_index[1]
        target_nodes = edge_index[0]

        h_i = h[target_nodes]
        h_j = h[neighbors_nodes]

        h_concat = torch.cat([h_i, h_j], dim=-1)

        eij = F.leaky_relu(
            (h_concat * self.attn).sum(dim=-1),
            negative_slope=self.scaling,
        )

        attention_score = self.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )

        message = attention_score.unsqueeze(-1) * h_j

        return message, h, target_nodes

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> Float[torch.Tensor, "nodes node_features"]:
        """Performs the forward pass.

        Args:
            node_features: input features for the nodes.
            edge_index: a matrix indicating which nodes are connected.
            COO format (i.e. 2, N_EDGES).
        Returns:
            Updated node features after attention.

        """
        message, transformed_node_features, target_nodes = (
            self.compute_attention(
                node_features=node_features,
                edge_index=edge_index,
            )
        )
        out = torch_scatter.scatter_add(
            message,
            target_nodes,
            dim=0,
            dim_size=transformed_node_features.size(0),
        )

        if self.add_skip_connection:
            out = out + transformed_node_features

        if self.concat:
            out = out.view(-1, self.num_heads * self.n_hidden_features)

        else:
            out = torch.mean(out, dim=1)

        if self.apply_act:
            out = F.elu(out)

        return out


@jt(typechecker=typechecker)
class MultiHeadEdgeGATLayer(nn.Module):
    """Implements a multihead graph attention layer.

    Attributes
        n_node_features: number of input node features.
        n_hidden_features: number of hidden features in intermediate layers.
        scaling: scaling constant for LeakyRelu
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_hidden_features: int,
        dropout: float,
        scaling: float = 0.2,
        num_heads: int = 8,
        apply_act: bool = True,
    ):
        super().__init__()

        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_hidden_features = n_hidden_features
        self.head_dimension = n_hidden_features // num_heads
        self.num_heads = num_heads
        self.dropout = dropout
        self.scaling = scaling
        self.apply_act = apply_act
        self.multiheadgat = self._get_attention_heads()

    def _get_attention_heads(self) -> nn.ModuleList:
        attention_heads = []
        for i in range(self.num_heads):
            attention_heads.append(
                GraphAttentionLayerEdge(
                    n_node_features=self.n_node_features,
                    n_edge_features=self.n_edge_features,
                    n_hidden_features=self.head_dimension,
                    dropout=self.dropout,
                    scaling=self.scaling,
                    apply_act=self.apply_act,
                ),
            )
        return nn.ModuleList(attention_heads)

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "nodes node_features"],
        Float[torch.Tensor, "edges edge_features"],
    ]:
        heads_nodes_out = []
        heads_edges_out = []

        for attn_head in self.multiheadgat:
            n_out, e_out = attn_head(
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index,
            )
            heads_edges_out.append(e_out)
            heads_nodes_out.append(n_out)

        return torch.cat(heads_nodes_out, dim=-1), torch.cat(
            heads_edges_out,
            dim=-1,
        )


@jt(typechecker=typechecker)
class EmbeddingGATEdge(nn.Module):
    """Implements a multi-head graph attention layer with edge features.

    This layer supports using discrete node and edge feature indices that
    are first converted to embeddings before performing multi-head attention.
    The attention mechanism incorporates both node and edge features to
    compute attention scores and updated node features.

    Attributes:
        n_node_features: Number of input node features.
        n_hidden_features: Number of hidden features in intermediate layers.
        scaling: Scaling constant for the LeakyReLU activation.
    """

    def __init__(
        self,
        n_node_dict: int,
        n_edge_dict: int,
        embedding_dim: int,
        n_hidden_features: int,
        num_heads: int = 1,
        scaling: float = 0.2,
        dropout: float = 0.0,
        apply_act: bool = False,
        concat: bool = True,
        add_skip_connection: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.scaling = scaling
        self.concat = concat
        self.apply_act = apply_act
        self.num_heads = num_heads
        self.n_hidden_features = n_hidden_features
        self.dropout = nn.Dropout(p=dropout)
        self.add_skip_connection = add_skip_connection

        self.batch_norm = nn.Identity()

        if batch_norm:
            self.batch_norm = nn.LayerNorm(n_hidden_features * num_heads)

        self.node_embedding = nn.Embedding(n_node_dict, embedding_dim)
        self.edge_embedding = nn.Embedding(n_edge_dict, embedding_dim)

        self.w = nn.Linear(embedding_dim, n_hidden_features * num_heads)
        self.edgew = nn.Linear(embedding_dim, n_hidden_features * num_heads)
        self.attn = nn.Parameter(
            torch.Tensor(1, num_heads, 3 * n_hidden_features),
        )
        nn.init.xavier_uniform_(self.attn.data)

    def compute_attention(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "attention_score 1"],  # noqa: F722
        Float[torch.Tensor, "nodes hidden_features"],  # noqa: F722
        Float[torch.Tensor, "edges hidden_features"],  # noqa: F722
        Int[torch.Tensor, " target_index"],
    ]:
        """Computes attention scores between connected nodes.

        The attention mechanism is defined as:

            α_ij = softmax_j( LeakyReLU( aᵀ [ W h_i || W h_j || W e_ij ] ) )

        where:
        - W is a learnable weight matrix,
        - a is a learnable attention vector,
        - || denotes vector concatenation,
        - e_ij represents edge features,
        - softmax_j is computed over all neighbors j ∈ N(i) of node i.

        The updated feature for node i is then:

            h_i' = σ( Σ_{j ∈ N(i)} α_ij · W h_j )

        where σ is a non-linear activation function.

        Args:
            node_features: Input node features with shape (N, F),
                where N is the number of nodes and F the number of features.
            edge_features: Input edge features with shape (E, F),
                where E is the number of edges.
            edge_index: Graph connectivity in COO format with shape (2, E),
                where the first row contains target node indices and
                the second row contains source node indices.

        Returns:
            A tuple containing:
                - Message after applying attention to node features.
                - Transformed node features.
                - Transformed edge features.
                - Indices of target nodes.
        """

        h = self.dropout(self.w(node_features)).view(
            -1,
            self.num_heads,
            self.n_hidden_features,
        )

        edge_h = self.edgew(edge_features).view(
            -1,
            self.num_heads,
            self.n_hidden_features,
        )

        neighbors_nodes = edge_index[1]
        target_nodes = edge_index[0]

        h_i = h[target_nodes]
        h_j = h[neighbors_nodes]

        h_concat = torch.cat([h_i, h_j, edge_h], dim=-1)

        eij = F.leaky_relu(
            (h_concat * self.attn).sum(dim=-1),
            negative_slope=self.scaling,
        )

        attention_score = self.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )

        message = attention_score.unsqueeze(-1) * h_j

        return message, h, edge_h, target_nodes

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "nodes node_features"],
        Float[torch.Tensor, "edges edge_features"],
    ]:
        if not torch.is_floating_point(
            node_features,
        ) and not torch.is_floating_point(edge_features):
            node_features = self.node_embedding(node_features)
            edge_features = self.edge_embedding(edge_features)

        (
            message,
            transformed_node_features,
            transformed_edge_features,
            target_nodes,
        ) = self.compute_attention(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )

        out = torch_scatter.scatter_add(
            message,
            target_nodes,
            dim=0,
            dim_size=transformed_node_features.size(0),
        )

        if self.add_skip_connection:
            out = out + transformed_node_features

        if self.concat:
            out = out.view(-1, self.num_heads * self.n_hidden_features)
            transformed_edge_features = transformed_edge_features.view(
                -1,
                self.num_heads * self.n_hidden_features,
            )

        else:
            out = torch.mean(out, dim=1)

        out = self.batch_norm(out)
        transformed_edge_features = self.batch_norm(transformed_edge_features)
        if self.apply_act:
            out = F.elu(out)

        return out, transformed_edge_features
