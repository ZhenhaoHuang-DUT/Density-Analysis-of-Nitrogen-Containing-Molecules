"""
alignn_model.py

ALIGNN (Atomistic Line Graph Neural Network) implementation
Paper: "Atomistic Line Graph Neural Network for improved materials property predictions"
Fixed version: ensure dimension consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter_mean


class EdgeGatedGraphConv(MessagePassing):
    """Edge-gated graph convolution layer used in ALIGNN"""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')

        # store dimension info privately (do not let parent class use them)
        self._node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # combined feature dimension for message computation
        combined_dim = node_dim * 2 + edge_dim

        # node update network
        self.node_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # edge gating network
        self.edge_gate_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # edge update network
        self.edge_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # update node features (residual)
        x_new = x + out

        # update edge features using updated node features
        edge_attr_new = self.update_edge(x_new, edge_index, edge_attr)

        return x_new, edge_attr_new

    def message(self, x_i, x_j, edge_attr):
        # concatenate source node, target node and edge features
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # compute edge gate
        gate = self.edge_gate_net(combined)

        # compute message
        message = self.node_net(combined)

        # apply gate to message
        return gate * message

    def update_edge(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_i, x_j = x[row], x[col]

        # compute updated edge features
        edge_combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        edge_update = self.edge_net(edge_combined)

        return edge_attr + edge_update


class ALIGNNLayer(nn.Module):
    """Single ALIGNN layer: atomic graph conv + line graph conv"""

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        # atomic graph convolution
        self.atom_conv = EdgeGatedGraphConv(node_dim, edge_dim, hidden_dim)

        # line-graph (edge) convolution; operates on edges
        self.edge_conv = EdgeGatedGraphConv(edge_dim, edge_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, line_graph_edge_index, line_graph_edge_attr):
        # 1) atomic graph convolution
        x_new, edge_attr_new = self.atom_conv(x, edge_index, edge_attr)

        # 2) line-graph convolution (operates on edge features)
        edge_attr_updated, _ = self.edge_conv(
            edge_attr_new,
            line_graph_edge_index,
            line_graph_edge_attr
        )

        return x_new, edge_attr_updated


class ALIGNN(nn.Module):
    """Full ALIGNN model"""

    def __init__(
            self,
            node_dim=9,
            edge_dim=6,
            line_edge_dim=5,
            global_dim=6,
            hidden_dim=256,
            num_layers=4,
            num_ffn_layers=2,
            dropout=0.1,
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.line_edge_dim = line_edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ----- input projection layers -----
        # project all features into hidden_dim
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        self.line_edge_embed = nn.Linear(line_edge_dim, hidden_dim)
        self.global_embed = nn.Linear(global_dim, hidden_dim)

        # ----- ALIGNN layers -----
        # inside ALIGNN layers we keep hidden_dim feature sizes
        self.alignn_layers = nn.ModuleList([
            ALIGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # skip (residual) connections per layer
        self.skip_connections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # ----- output network -----
        # after pooling we will concatenate pooled node features and global embedding => hidden_dim * 2
        ffn_input_dim = hidden_dim * 2

        ffn_layers = []
        current_dim = ffn_input_dim
        for _ in range(num_ffn_layers):
            ffn_layers.append(nn.Linear(current_dim, hidden_dim))
            ffn_layers.append(nn.SiLU())
            ffn_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.ffn = nn.Sequential(*ffn_layers)

        # final output MLP
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data):
        # extract data fields
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        line_edge_index = data.line_graph_edge_index
        line_edge_attr = data.line_graph_edge_attr
        u = data.u
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # embed features
        x = self.node_embed(x)  # (N, node_dim) -> (N, hidden_dim)
        edge_attr = self.edge_embed(edge_attr)  # (E, edge_dim) -> (E, hidden_dim)
        line_edge_attr = self.line_edge_embed(line_edge_attr)  # (L, line_edge_dim) -> (L, hidden_dim)

        # process global features: ensure u is (batch_size, global_dim)
        if u.dim() == 1:
            u = u.unsqueeze(0)
        elif u.dim() == 2 and u.size(0) != batch.max().item() + 1:
            # if global features are per-node, pool to graph level
            u = global_mean_pool(u, batch)

        u_embed = self.global_embed(u)  # (batch_size, hidden_dim)

        # ALIGNN message passing with skip connections
        x_skip = x
        for i, layer in enumerate(self.alignn_layers):
            x_new, edge_attr_new = layer(
                x, edge_index, edge_attr,
                line_edge_index, line_edge_attr
            )

            x = x_new + self.skip_connections[i](x_skip)
            edge_attr = edge_attr_new
            x_skip = x

        # global pooling
        x_pool = global_mean_pool(x, batch)  # (batch_size, hidden_dim)

        # concatenate pooled node features and global embedding
        x_combined = torch.cat([x_pool, u_embed], dim=-1)  # (batch_size, hidden_dim*2)

        # FFN processing
        x_processed = self.ffn(x_combined)

        # final output
        output = self.output_layer(x_processed)  # (batch_size, 1)

        return output


class ALIGNNWithEnergyReadout(ALIGNN):
    """ALIGNN variant with an additional energy readout head"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.energy_readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, data):
        # base ALIGNN forward steps
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        line_edge_index = data.line_graph_edge_index
        line_edge_attr = data.line_graph_edge_attr
        u = data.u
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        line_edge_attr = self.line_edge_embed(line_edge_attr)

        if u.dim() == 1:
            u = u.unsqueeze(0)
        u_embed = self.global_embed(u)

        x_skip = x
        for i, layer in enumerate(self.alignn_layers):
            x_new, edge_attr_new = layer(
                x, edge_index, edge_attr,
                line_edge_index, line_edge_attr
            )
            x = x_new + self.skip_connections[i](x_skip)
            edge_attr = edge_attr_new
            x_skip = x

        x_pool = global_mean_pool(x, batch)
        x_combined = torch.cat([x_pool, u_embed], dim=-1)
        x_processed = self.ffn(x_combined)
        density = self.output_layer(x_processed)

        energy = self.energy_readout(x_pool)

        return density, energy


class SimpleALIGNN(nn.Module):
    """Simplified ALIGNN for quick testing"""

    def __init__(
            self,
            node_dim=9,
            edge_dim=6,
            line_edge_dim=5,
            global_dim=6,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1,
    ):
        super().__init__()

        # projection layers
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        self.global_embed = nn.Linear(global_dim, hidden_dim)

        # simplified ALIGNN layers
        self.alignn_layers = nn.ModuleList([
            ALIGNNLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # output MLP
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.node_embed(data.x)
        edge_attr = self.edge_embed(data.edge_attr)
        u = data.u.unsqueeze(0) if data.u.dim() == 1 else data.u
        u_embed = self.global_embed(u)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # simplified message passing
        for layer in self.alignn_layers:
            x, edge_attr = layer(
                x, data.edge_index, edge_attr,
                data.line_graph_edge_index, data.line_graph_edge_attr
            )

        # pooling and output
        x_pool = global_mean_pool(x, batch)
        x_combined = torch.cat([x_pool, u_embed], dim=-1)
        output = self.output_layer(x_combined)

        return output


if __name__ == "__main__":
    # test script
    print("Testing ALIGNN model...")

    from torch_geometric.data import Data

    # create synthetic test data
    test_data = Data(
        x=torch.randn(25, 9),
        edge_index=torch.randint(0, 25, (2, 48)),
        edge_attr=torch.randn(48, 6),
        line_graph_edge_index=torch.randint(0, 48, (2, 74)),
        line_graph_edge_attr=torch.randn(74, 5),
        u=torch.randn(1, 6),
        y=torch.randn(1, 1)
    )

    model = ALIGNN(
        node_dim=9,
        edge_dim=6,
        line_edge_dim=5,
        global_dim=6,
        hidden_dim=256,
        num_layers=2,
        num_ffn_layers=1,
        dropout=0.0
    )

    output = model(test_data)
    print("Input shapes:")
    print(f"  x: {test_data.x.shape}")
    print(f"  edge_index: {test_data.edge_index.shape}")
    print(f"  edge_attr: {test_data.edge_attr.shape}")
    print(f"  line_graph_edge_index: {test_data.line_graph_edge_index.shape}")
    print(f"  line_graph_edge_attr: {test_data.line_graph_edge_attr.shape}")
    print(f"  u: {test_data.u.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Test completed successfully.")