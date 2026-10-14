
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.layers import TransformerEncoder


# ============================================================
# Graph relaxation for the auxiliary property head
# ============================================================

def relax_graph(node_logits, edge_logits, mode="st"):
    """
    Convert generator logits into the graph representation that is shown
    to the auxiliary property head.

    The representation is aligned with how generated graphs are decoded
    by DrugGEN's matrices2mol():

        atom type = argmax over node logits
        bond type = argmax over edge logits

    Only the strictly lower triangle is used for chemical bonds.
    The result is then mirrored to maintain graph symmetry.

    Modes
    -----
    st:
        Straight-through discrete representation.

        Forward pass:
            exactly the discrete one-hot graph corresponding to argmax.

        Backward pass:
            uses the softmax gradient.

    softmax:
        Differentiable probability representation.

    logits:
        Raw logits.

    Returns
    -------
    edge:
        [B, V, V, E]

    node:
        [B, V, N]
    """

    if mode == "logits":
        return edge_logits, node_logits

    if mode not in ("st", "softmax"):
        raise ValueError(
            f"Unknown property_input mode: {mode}"
        )

    node = F.softmax(
        node_logits,
        dim=-1,
    )

    edge = F.softmax(
        edge_logits,
        dim=-1,
    )

    if mode == "st":

        node_hard = F.one_hot(
            node.argmax(dim=-1),
            node.shape[-1],
        ).to(node.dtype)

        edge_hard = F.one_hot(
            edge.argmax(dim=-1),
            edge.shape[-1],
        ).to(edge.dtype)

        node = (
            node_hard
            + node
            - node.detach()
        )

        edge = (
            edge_hard
            + edge
            - edge.detach()
        )

    # --------------------------------------------------------
    # Keep only the strictly lower triangle.
    #
    # DrugGEN's decoder reads the lower triangle for bonds.
    # --------------------------------------------------------

    V = edge.shape[1]

    lower = torch.tril(
        torch.ones(
            V,
            V,
            device=edge.device,
            dtype=edge.dtype,
        ),
        diagonal=-1,
    ).view(
        1,
        V,
        V,
        1,
    )

    edge = edge * lower

    # Restore symmetry.
    edge = (
        edge
        + edge.transpose(1, 2)
    )

    # --------------------------------------------------------
    # Diagonal = no bond.
    #
    # DrugGEN bond class 0 is the no-bond class.
    # --------------------------------------------------------

    eye = torch.eye(
        V,
        device=edge.device,
        dtype=edge.dtype,
    ).view(
        1,
        V,
        V,
        1,
    )

    no_bond = torch.zeros(
        1,
        1,
        1,
        edge.shape[-1],
        device=edge.device,
        dtype=edge.dtype,
    )

    no_bond[..., 0] = 1.0

    edge = (
        edge
        + eye * no_bond
    )

    return edge, node


# ============================================================
# Graph-aware auxiliary property head
# ============================================================

class PropertyHead(nn.Module):
    """
    Graph-aware, permutation-invariant regression head.

    Input
    -----
    node:
        [B, V, dim]

    edge:
        [B, V, V, dim]

    Output
    ------
    [normalized pIC50, normalized logBB]

    The head is deliberately condition-free.

    It receives graph representations rather than the requested
    condition so that it cannot simply copy the requested pIC50
    or logBB value.

    Node and edge representations are separately transformed,
    normalized and mean-pooled before being combined.
    """

    def __init__(
        self,
        dim,
        act,
        n_props=2,
        hidden=128,
        dropout=0.1,
    ):
        super().__init__()

        self.node_phi = nn.Sequential(
            nn.Linear(
                dim,
                dim,
            ),
            act,
        )

        self.edge_phi = nn.Sequential(
            nn.Linear(
                dim,
                dim,
            ),
            act,
        )

        self.node_norm = nn.LayerNorm(
            dim
        )

        self.edge_norm = nn.LayerNorm(
            dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                2 * dim,
                hidden,
            ),
            act,

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden,
                64,
            ),
            act,

            nn.Linear(
                64,
                n_props,
            ),
        )

    def forward(
        self,
        node,
        edge,
    ):

        node_pool = self.node_norm(
            self.node_phi(node).mean(
                dim=1
            )
        )

        edge_pool = self.edge_norm(
            self.edge_phi(edge).mean(
                dim=(1, 2)
            )
        )

        combined = torch.cat(
            [
                node_pool,
                edge_pool,
            ],
            dim=-1,
        )

        return self.mlp(
            combined
        )


# ============================================================
# Conditional DrugGEN Generator
# ============================================================

class ConditionalGenerator(nn.Module):
    """
    Conditional DrugGEN Generator.

    Condition vector:

        [
            target_condition,
            normalized_pIC50,
            normalized_logBB
        ]

    The original DrugGEN node/edge MLPs and Transformer are retained.

    Condition injection occurs at two locations:

    1. Before the Transformer.
    2. Directly before the final categorical atom/bond readout.

    The second pathway is the Issue-3 chemistry-level conditioning
    mechanism. It gives the requested continuous properties a direct
    route to the final categorical chemical decisions.
    """

    def __init__(
        self,
        act,
        vertexes,
        edges,
        nodes,
        dropout,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
        condition_dim=3,
    ):
        super(
            ConditionalGenerator,
            self
        ).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.condition_dim = condition_dim

        # --------------------------------------------------------
        # Activation
        # --------------------------------------------------------

        if act == "relu":

            act = nn.ReLU()

        elif act == "leaky":

            act = nn.LeakyReLU()

        elif act == "sigmoid":

            act = nn.Sigmoid()

        elif act == "tanh":

            act = nn.Tanh()

        else:

            raise ValueError(
                f"Unsupported activation function: {act}"
            )

        # --------------------------------------------------------
        # Original DrugGEN dimensions
        # --------------------------------------------------------

        self.features = (
            vertexes
            * vertexes
            * edges
            + vertexes
            * nodes
        )

        self.transformer_dim = (
            vertexes
            * vertexes
            * dim
            + vertexes
            * dim
        )

        # --------------------------------------------------------
        # Original node processing
        # --------------------------------------------------------

        self.node_layers = nn.Sequential(
            nn.Linear(
                nodes,
                64,
            ),

            act,

            nn.Linear(
                64,
                dim,
            ),

            act,

            nn.Dropout(
                self.dropout
            ),
        )

        # --------------------------------------------------------
        # Original edge processing
        # --------------------------------------------------------

        self.edge_layers = nn.Sequential(
            nn.Linear(
                edges,
                64,
            ),

            act,

            nn.Linear(
                64,
                dim,
            ),

            act,

            nn.Dropout(
                self.dropout
            ),
        )

        # --------------------------------------------------------
        # Transformer-level condition pathway
        #
        # [BACE1 target,
        #  normalized pIC50,
        #  normalized logBB]
        #
        #        ↓
        #
        # Linear(3 -> dim)
        #
        #        ↓
        #
        #      Tanh
        # --------------------------------------------------------

        self.condition_layers = nn.Sequential(
            nn.Linear(
                condition_dim,
                dim,
            ),

            nn.Tanh(),
        )

        # --------------------------------------------------------
        # Transformer
        # --------------------------------------------------------

        self.TransformerEncoder = TransformerEncoder(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            act=act,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.dropout,
        )

        # --------------------------------------------------------
        # Issue-3 chemistry-level condition pathway
        #
        # This is a separate projection used immediately before
        # categorical atom/bond decisions.
        #
        # The Transformer output is concatenated with this
        # condition representation before readout.
        # --------------------------------------------------------

        self.readout_condition = nn.Sequential(
            nn.Linear(
                condition_dim,
                dim,
            ),

            nn.Tanh(),
        )

        # --------------------------------------------------------
        # Condition-aware categorical readouts
        #
        # Original:
        #
        #     dim -> nodes
        #     dim -> edges
        #
        # New:
        #
        #     [dim graph || dim condition]
        #                  ↓
        #             2 * dim
        #                  ↓
        #             atom / bond
        # --------------------------------------------------------

        self.readout_e = nn.Linear(
            self.dim * 2,
            edges,
        )

        self.readout_n = nn.Linear(
            self.dim * 2,
            nodes,
        )

        self.softmax = nn.Softmax(
            dim=-1
        )

    def forward(
        self,
        z_e,
        z_n,
        condition,
    ):
        """
        Forward pass.

        Parameters
        ----------
        z_e:
            Input edge graph tensor.

        z_n:
            Input node graph tensor.

        condition:
            [batch, 3]

            [
                BACE1 target condition,
                normalized pIC50,
                normalized logBB
            ]

        Returns
        -------
        node:
            Post-Transformer node representation.

        edge:
            Post-Transformer edge representation.

        node_sample:
            Atom logits.

        edge_sample:
            Bond logits.
        """

        # ========================================================
        # Original DrugGEN input projections
        # ========================================================

        node = self.node_layers(
            z_n
        )

        edge = self.edge_layers(
            z_e
        )

        # ========================================================
        # Transformer-level condition injection
        # ========================================================

        c = self.condition_layers(
            condition
        ).to(
            node.dtype
        )

        node = (
            node
            + c.unsqueeze(1)
        )

        edge = (
            edge
            + c.unsqueeze(1).unsqueeze(1)
        )

        # ========================================================
        # Preserve symmetric graph representation
        # ========================================================

        edge = (
            edge
            + edge.permute(
                0,
                2,
                1,
                3,
            )
        ) / 2

        # ========================================================
        # Transformer
        # ========================================================

        node, edge = self.TransformerEncoder(
            node,
            edge,
        )

        # ========================================================
        # ISSUE-3 CHEMISTRY-LEVEL CONDITION-AWARE READOUT
        #
        # The condition is injected AGAIN immediately before
        # categorical chemical decisions.
        #
        # This gives pIC50/logBB a direct path to:
        #
        #     Transformer representation
        #              +
        #        condition representation
        #              ↓
        #        categorical output
        # ========================================================

        readout_condition = (
            self.readout_condition(
                condition
            ).to(
                node.dtype
            )
        )

        # --------------------------------------------------------
        # Node condition
        #
        # [B, dim]
        #     ↓
        # [B, V, dim]
        # --------------------------------------------------------

        node_condition = (
            readout_condition
            .unsqueeze(1)
            .expand(
                -1,
                node.shape[1],
                -1,
            )
        )

        # --------------------------------------------------------
        # Edge condition
        #
        # [B, dim]
        #     ↓
        # [B, V, V, dim]
        # --------------------------------------------------------

        edge_condition = (
            readout_condition
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(
                -1,
                edge.shape[1],
                edge.shape[2],
                -1,
            )
        )

        # --------------------------------------------------------
        # Concatenate graph representation + condition
        # --------------------------------------------------------

        node_readout = torch.cat(
            [
                node,
                node_condition,
            ],
            dim=-1,
        )

        edge_readout = torch.cat(
            [
                edge,
                edge_condition,
            ],
            dim=-1,
        )

        # ========================================================
        # Final categorical decisions
        # ========================================================

        node_sample = self.readout_n(
            node_readout
        )

        edge_sample = self.readout_e(
            edge_readout
        )

        return (
            node,
            edge,
            node_sample,
            edge_sample,
        )


# ============================================================
# Conditional DrugGEN Discriminator
# ============================================================

class ConditionalDiscriminator(nn.Module):
    """
    Conditional DrugGEN Discriminator with an auxiliary continuous
    condition-prediction head.

    Main critic output:
        real/fake Wasserstein score

    Auxiliary output:
        [normalized_pIC50, normalized_logBB]

    Issue 3:
        The auxiliary property head is graph-aware and condition-free.

    It processes the graph through a separate Transformer pass WITHOUT
    condition injection, preventing the head from simply reading the
    requested condition back from the input.
    """

    def __init__(
        self,
        act,
        vertexes,
        edges,
        nodes,
        dropout,
        dim=128,
        depth=1,
        heads=8,
        mlp_ratio=3,
        condition_dim=3,
        property_dim=2,
        property_dropout=0.1,
    ):
        super(
            ConditionalDiscriminator,
            self
        ).__init__()

        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes

        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self.condition_dim = condition_dim
        self.property_dim = property_dim

        # --------------------------------------------------------
        # Activation
        # --------------------------------------------------------

        if act == "relu":

            act = nn.ReLU()

        elif act == "leaky":

            act = nn.LeakyReLU()

        elif act == "sigmoid":

            act = nn.Sigmoid()

        elif act == "tanh":

            act = nn.Tanh()

        else:

            raise ValueError(
                f"Unsupported activation function: {act}"
            )

        # --------------------------------------------------------
        # Original DrugGEN dimensions
        # --------------------------------------------------------

        self.features = (
            vertexes
            * vertexes
            * edges
            + vertexes
            * nodes
        )

        self.transformer_dim = (
            vertexes
            * vertexes
            * dim
            + vertexes
            * dim
        )

        # --------------------------------------------------------
        # Node processing
        # --------------------------------------------------------

        self.node_layers = nn.Sequential(
            nn.Linear(
                nodes,
                64,
            ),

            act,

            nn.Linear(
                64,
                dim,
            ),

            act,

            nn.Dropout(
                self.dropout
            ),
        )

        # --------------------------------------------------------
        # Edge processing
        # --------------------------------------------------------

        self.edge_layers = nn.Sequential(
            nn.Linear(
                edges,
                64,
            ),

            act,

            nn.Linear(
                64,
                dim,
            ),

            act,

            nn.Dropout(
                self.dropout
            ),
        )

        # --------------------------------------------------------
        # Condition pathway
        # --------------------------------------------------------

        self.condition_layers = nn.Sequential(
            nn.Linear(
                condition_dim,
                dim,
            ),

            nn.Tanh(),
        )

        # --------------------------------------------------------
        # Shared Transformer
        # --------------------------------------------------------

        self.TransformerEncoder = TransformerEncoder(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            act=act,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.dropout,
        )

        # --------------------------------------------------------
        # Original critic
        # --------------------------------------------------------

        self.node_features = (
            vertexes
            * dim
        )

        self.edge_features = (
            vertexes
            * vertexes
            * dim
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(
                self.node_features,
                64,
            ),

            act,

            nn.Linear(
                64,
                32,
            ),

            act,

            nn.Linear(
                32,
                16,
            ),

            act,

            nn.Linear(
                16,
                1,
            ),
        )

        # --------------------------------------------------------
        # Issue-3 graph-aware property head
        #
        # It predicts:
        #
        #     normalized pIC50
        #     normalized logBB
        #
        # from graph structure alone.
        # --------------------------------------------------------

        self.property_head = PropertyHead(
            dim=self.dim,
            act=act,
            n_props=property_dim,
            dropout=property_dropout,
        )

    def predict_properties(
        self,
        z_e,
        z_n,
    ):
        """
        Predict:

            [normalized pIC50,
             normalized logBB]

        from the graph alone.

        IMPORTANT:
        No condition is injected in this path.
        """

        # --------------------------------------------------------
        # Raw graph -> latent representation
        # --------------------------------------------------------

        node = self.node_layers(
            z_n
        )

        edge = self.edge_layers(
            z_e
        )

        # --------------------------------------------------------
        # Symmetric edge representation
        # --------------------------------------------------------

        edge = (
            edge
            + edge.permute(
                0,
                2,
                1,
                3,
            )
        ) / 2

        # --------------------------------------------------------
        # Condition-free Transformer pass
        # --------------------------------------------------------

        node, edge = self.TransformerEncoder(
            node,
            edge,
        )

        # --------------------------------------------------------
        # Graph-aware property prediction
        # --------------------------------------------------------

        return self.property_head(
            node,
            edge,
        )

    def forward(
        self,
        z_e,
        z_n,
        condition,
        return_property=True,
    ):
        """
        Returns
        -------
        critic_score:
            [batch, 1]

        property_prediction:
            [batch, 2]

            or None when return_property=False.

        property_prediction:

            [
                normalized pIC50,
                normalized logBB
            ]
        """

        b = z_n.shape[0]

        # ========================================================
        # Graph projection
        # ========================================================

        node = self.node_layers(
            z_n
        )

        edge = self.edge_layers(
            z_e
        )

        # ========================================================
        # Condition injection for critic path
        # ========================================================

        c = self.condition_layers(
            condition
        ).to(
            node.dtype
        )

        node = (
            node
            + c.unsqueeze(1)
        )

        edge = (
            edge
            + c.unsqueeze(1).unsqueeze(1)
        )

        # ========================================================
        # Symmetric graph
        # ========================================================

        edge = (
            edge
            + edge.permute(
                0,
                2,
                1,
                3,
            )
        ) / 2

        # ========================================================
        # Transformer critic path
        # ========================================================

        node, edge = self.TransformerEncoder(
            node,
            edge,
        )

        # ========================================================
        # Critic
        # ========================================================

        node = node.view(
            b,
            -1,
        )

        critic_score = self.node_mlp(
            node
        )

        # ========================================================
        # Property prediction
        #
        # This intentionally uses the separate condition-free
        # graph path.
        # ========================================================

        property_prediction = None

        if return_property:

            property_prediction = (
                self.predict_properties(
                    z_e,
                    z_n,
                )
            )

        return (
            critic_score,
            property_prediction,
        )
