"""Graph attention layers"""

import torch
import torch.nn.functional as F  # noqa: N812
import torch_scatter
from jaxtyping import Float, Int
from jaxtyping import jaxtyped as jt
from torch import nn
from typeguard import typechecked as typechecker


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
        n_out_features: Number of hidden features.
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
        n_input_features: int,
        n_out_features: int,
        dropout: float = 0.0,
        scaling: float = 0.2,
        num_heads: int = 8,
        apply_act: bool = True,
        concat: bool = True,
        add_skip_connection: bool = True,
        batch_norm: bool = False,
        add_bias: bool = False,
        share_weights: bool = True,
    ):
        super().__init__()

        self.scaling = scaling
        self.concat = concat
        self.apply_act = apply_act
        self.num_heads = num_heads
        self.n_out_features = n_out_features
        self.dropout = nn.Dropout(p=dropout)
        self.add_skip_connection = add_skip_connection
        self.batch_norm = batch_norm

        if batch_norm:
            if concat:
                self.norm = nn.LayerNorm(n_out_features * num_heads)
            else:
                self.norm = nn.LayerNorm(n_out_features)

        self.target_w = nn.Linear(n_input_features, n_out_features * num_heads)

        if share_weights:
            self.neighbor_w = self.target_w
        else:
            self.neighbor_w = nn.Linear(
                n_input_features,
                n_out_features * num_heads,
            )

        self.attn = nn.Linear(2 * n_out_features, 1, bias=add_bias)

    def compute_attention(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "attention_score 1"],  # noqa: F722
        Float[torch.Tensor, "nodes hidden_features"],  # noqa: F722
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
                where E is the number of edges. The first row contains source
                node indices, and the second row contains target node indices.
        Returns:
            Attention scores for nodes.
        """
        neighbors_nodes = edge_index[0]
        target_nodes = edge_index[1]

        h_i = self.target_w(node_features[target_nodes]).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        h_j = self.neighbor_w(node_features[neighbors_nodes]).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        h_concat = torch.cat([h_i, h_j], dim=-1)

        eij = F.leaky_relu(
            self.attn(h_concat),
            negative_slope=self.scaling,
        )

        attention_score = self.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )

        message = attention_score * h_j

        return message, h_i

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
        (
            message,
            transformed_node_features,
        ) = self.compute_attention(
            node_features=node_features,
            edge_index=edge_index,
        )

        if self.add_skip_connection:
            message = message + transformed_node_features

        out = torch_scatter.scatter_add(
            message,
            edge_index[1],
            dim=0,
            dim_size=node_features.size(0),
        )

        if self.concat:
            out = out.view(-1, self.num_heads * self.n_out_features)

        else:
            out = torch.mean(out, dim=1)

        if self.batch_norm:
            out = self.norm(out)

        if self.apply_act:
            out = F.elu(out)

        return out


@jt(typechecker=typechecker)
class MultiHeadEdgeGATLayer(nn.Module):
    """Implements a multihead graph attention layer.

    Attributes
        n_node_features: number of input node features.
        n_out_features: number of hidden features in intermediate layers.
        scaling: scaling constant for LeakyRelu
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        n_out_features: int,
        dropout: float = 0.0,
        scaling: float = 0.2,
        num_heads: int = 8,
        apply_act: bool = True,
        concat: bool = True,
        add_skip_connection: bool = True,
        batch_norm: bool = True,
        share_weights: bool = True,
    ):
        super().__init__()

        self.scaling = scaling
        self.concat = concat
        self.apply_act = apply_act
        self.num_heads = num_heads
        self.n_out_features = n_out_features
        self.dropout = nn.Dropout(p=dropout)
        self.add_skip_connection = add_skip_connection

        self.batch_norm = nn.Identity()

        if batch_norm:
            if concat:
                self.batch_norm = nn.LayerNorm(n_out_features * num_heads)
            else:
                self.batch_norm = nn.LayerNorm(n_out_features)

        self.target_w = nn.Linear(n_node_features, n_out_features * num_heads)

        if share_weights:
            self.neighbor_w = self.target_w
        else:
            self.neighbor_w = nn.Linear(
                n_node_features,
                n_out_features * num_heads,
            )
        self.edgew = nn.Linear(n_edge_features, n_out_features * num_heads)
        self.attn = nn.Linear(3 * n_out_features, 1)

    def compute_attention(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "attention_score 1"],  # noqa: F722
        Float[torch.Tensor, "nodes hidden_features"],  # noqa: F722
        Float[torch.Tensor, "edges hidden_features"],  # noqa: F722
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
                where the first row contains source node indices and
                the second row contains target node indices.

        Returns:
            A tuple containing:
                - Message after applying attention to node features.
                - Transformed node features.
                - Transformed edge features.
                - Indices of target nodes.
        """

        edge_h = self.edgew(edge_features).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        neighbors_nodes = edge_index[0]
        target_nodes = edge_index[1]

        h_i = self.target_w(node_features[target_nodes]).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        h_j = self.neighbor_w(node_features[neighbors_nodes]).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        h_concat = torch.cat([h_i, h_j, edge_h], dim=-1)

        eij = F.leaky_relu(
            self.attn(h_concat),
            negative_slope=self.scaling,
        )

        attention_score = self.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )

        message = attention_score * h_j

        return message, h_i, edge_h

    def forward(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "nodes node_features"],
        Float[torch.Tensor, "edges edge_features"],
    ]:
        (
            message,
            transformed_node_features,
            transformed_edge_features,
        ) = self.compute_attention(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )

        if self.add_skip_connection:
            message = message + transformed_node_features

        out = torch_scatter.scatter_add(
            message,
            edge_index[1],
            dim=0,
            dim_size=node_features.size(0),
        )

        if self.concat:
            out = out.view(-1, self.num_heads * self.n_out_features)
            transformed_edge_features = transformed_edge_features.view(
                -1,
                self.num_heads * self.n_out_features,
            )

        else:
            out = torch.mean(out, dim=1)
            transformed_edge_features = torch.mean(
                transformed_edge_features,
                dim=1,
            )

        out = self.batch_norm(out)
        transformed_edge_features = self.batch_norm(transformed_edge_features)
        if self.apply_act:
            out = F.elu(out)

        return out, transformed_edge_features


@jt(typechecker=typechecker)
class EmbeddingGATEdge(nn.Module):
    """Implements a multi-head graph attention layer with edge features.

    This layer supports using discrete node and edge feature indices that
    are first converted to embeddings before performing multi-head attention.
    The attention mechanism incorporates both node and edge features to
    compute attention scores and updated node features.

    Attributes:
        n_node_features: Number of input node features.
        n_out_features: Number of hidden features in intermediate layers.
        scaling: Scaling constant for the LeakyReLU activation.
    """

    def __init__(
        self,
        n_node_dict: int,
        n_edge_dict: int,
        embedding_dim: int,
        n_out_features: int,
        num_heads: int = 1,
        scaling: float = 0.2,
        dropout: float = 0.0,
        apply_act: bool = False,
        concat: bool = True,
        add_skip_connection: bool = True,
        batch_norm: bool = True,
        share_weights: bool = True,
    ):
        super().__init__()
        self.scaling = scaling
        self.concat = concat
        self.apply_act = apply_act
        self.num_heads = num_heads
        self.n_out_features = n_out_features
        self.dropout = nn.Dropout(p=dropout)
        self.add_skip_connection = add_skip_connection

        self.batch_norm = nn.Identity()
        if batch_norm:
            if concat:
                self.batch_norm = nn.LayerNorm(n_out_features * num_heads)
            else:
                self.batch_norm = nn.LayerNorm(n_out_features)

        self.node_embedding = nn.Embedding(n_node_dict, embedding_dim)
        self.edge_embedding = nn.Embedding(n_edge_dict, embedding_dim)

        self.target_w = nn.Linear(embedding_dim, n_out_features * num_heads)

        if share_weights:
            self.neighbor_w = self.target_w
        else:
            self.neighbor_w = nn.Linear(
                embedding_dim,
                n_out_features * num_heads,
            )

        self.edgew = nn.Linear(embedding_dim, n_out_features * num_heads)
        self.attn = nn.Linear(3 * n_out_features, 1)

    def compute_attention(
        self,
        node_features: Float[torch.Tensor, "nodes node_features"],
        edge_features: Float[torch.Tensor, "edges edge_features"],
        edge_index: Int[torch.Tensor, "2 edges"],
    ) -> tuple[
        Float[torch.Tensor, "attention_score 1"],  # noqa: F722
        Float[torch.Tensor, "nodes hidden_features"],  # noqa: F722
        Float[torch.Tensor, "edges hidden_features"],  # noqa: F722
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
                where the first row contains source node indices and
                the second row contains target node indices.

        Returns:
            A tuple containing:
                - Message after applying attention to node features.
                - Transformed node features.
                - Transformed edge features.
        """

        edge_h = self.edgew(edge_features).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        neighbors_nodes = edge_index[0]
        target_nodes = edge_index[1]

        h_i = self.target_w(node_features[target_nodes]).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        h_j = self.neighbor_w(node_features[neighbors_nodes]).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        h_concat = torch.cat([h_i, h_j, edge_h], dim=-1)

        eij = F.leaky_relu(
            self.attn(h_concat),
            negative_slope=self.scaling,
        )

        attention_score = self.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )

        message = attention_score * h_j

        return message, h_i, edge_h

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
        ) = self.compute_attention(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
        )

        if self.add_skip_connection:
            message = message + transformed_node_features

        out = torch_scatter.scatter_add(
            message,
            edge_index[1],
            dim=0,
            dim_size=node_features.size(0),
        )

        if self.concat:
            out = out.view(-1, self.num_heads * self.n_out_features)
            transformed_edge_features = transformed_edge_features.view(
                -1,
                self.num_heads * self.n_out_features,
            )

        else:
            out = torch.mean(out, dim=1)

        out = self.batch_norm(out)
        transformed_edge_features = self.batch_norm(transformed_edge_features)
        if self.apply_act:
            out = F.elu(out)

        return out, transformed_edge_features


@jt(typechecker=typechecker)
class EmbeddingGATEdgeV2(nn.Module):
    """Implements a multihead graph attention layer with edge features.

    This layer only supports categorical node and edge features.

    Attributes
        n_node_features: number of input node features.
        n_out_features: number of hidden features in intermediate layers.
        scaling: scaling constant for LeakyRelu
    """

    def __init__(
        self,
        n_node_dict: int,
        n_edge_dict: int,
        embedding_dim: int,
        n_out_features: int,
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
        self.n_out_features = n_out_features
        self.dropout = nn.Dropout(p=dropout)
        self.add_skip_connection = add_skip_connection

        self.batch_norm = nn.Identity()

        if batch_norm:
            self.batch_norm = nn.LayerNorm(n_out_features * num_heads)

        self.node_embedding = nn.Embedding(n_node_dict, embedding_dim)
        self.edge_embedding = nn.Embedding(n_edge_dict, embedding_dim)

        self.w = nn.Linear(2 * embedding_dim, n_out_features * num_heads)
        self.edgew = nn.Linear(embedding_dim, n_out_features * num_heads)
        self.attn = nn.Linear(2 * n_out_features, 1)

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
                where E is the number of edges. The first row contains source
                node indices, and the second row contains target node indices.
        Returns:
            Attention scores for nodes.
        """

        edge_h = self.edgew(edge_features).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        neighbors_nodes = edge_index[0]
        target_nodes = edge_index[1]

        h_i = node_features[target_nodes]
        h_j = node_features[neighbors_nodes]

        h = self.dropout(self.w(torch.cat([h_i, h_j], dim=-1))).view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        h_concat = F.leaky_relu(
            torch.cat([h, edge_h], dim=-1),
            negative_slope=self.scaling,
        )

        eij = self.attn(h_concat)

        attention_score = self.dropout(
            torch_scatter.scatter_softmax(
                src=eij,
                index=target_nodes,
                dim=0,
            ),
        )
        message = attention_score * h_j.view(
            -1,
            self.num_heads,
            self.n_out_features,
        )

        return (
            message,
            h_i.view(-1, self.num_heads, self.n_out_features),
            edge_h,
            target_nodes,
        )

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

        if self.add_skip_connection:
            message = message + transformed_node_features

        out = torch_scatter.scatter_add(
            message,
            target_nodes,
            dim=0,
            dim_size=node_features.size(0),
        )

        if self.concat:
            out = out.view(-1, self.num_heads * self.n_out_features)
            transformed_edge_features = transformed_edge_features.view(
                -1,
                self.num_heads * self.n_out_features,
            )

        else:
            out = torch.mean(out, dim=1)

        out = self.batch_norm(out)
        transformed_edge_features = self.batch_norm(transformed_edge_features)
        if self.apply_act:
            out = F.elu(out)

        return out, transformed_edge_features
