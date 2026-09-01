from . import ncut
import numpy as np
import torch
import dgl


class Molecule:
    """
    A molecule object contains the following attributes:
        ; molecule.num_atom : nb of atoms, an integer (N)
        ; molecule.atom_type : pytorch tensor of size N, each element is an atom type, an integer between 0 and num_atom_type-1
        ; molecule.atom_type_pe : pytorch tensor of size N, each element is an atom type positional encoding, an integer between 0 and num_atom-1
        ; molecule.bond_type : pytorch tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type-1 
        ; molecule.bag_of_atoms : pytorch tensor of size num_atom_type, histogram of atoms in the molecule
        ; molecule.logP_SA_cycle_normalized : the chemical property to regress, a pytorch float variable
        ; molecule.smile : the smile representation of the molecule for rdkit, a string   
    """
    def __init__(self, num_atom, num_atom_type):
        self.num_atom       = num_atom
        self.atom_type      = torch.zeros( num_atom , dtype=torch.long )
        self.atom_type_pe   = torch.zeros( num_atom , dtype=torch.long )
        self.bond_type      = torch.zeros( num_atom , num_atom, dtype=torch.long )
        self.bag_of_atoms   = torch.zeros( num_atom_type, dtype=torch.long)
        self.logP_SA        = torch.zeros( 1, dtype=torch.float)
        self.logP_SA_cycle_normalized  = torch.zeros( 1, dtype=torch.float)
        self.smile  = ''
    def set_bag_of_atoms(self):
        for tp in self.atom_type:
                self.bag_of_atoms[tp.item()] += 1
    def set_atom_type_pe(self):
        histogram={}
        for idx, tp in enumerate(self.atom_type):
            tpp=tp.item()
            if tpp not in histogram:
                histogram[tpp] = 0
            else:
                histogram[tpp] += 1
            self.atom_type_pe[idx] = histogram[tpp]
    def shuffle_indexing(self):
        idx = torch.randperm(self.num_atom)
        self.atom_type = self.atom_type[idx]
        self.atom_type_pe = self.atom_type_pe[idx]
        self.bond_type = self.bond_type[idx][:,idx]
        return idx
    def __len__(self):
        return self.num_atom
    

def compute_ncut(Adj, R):
    # Apply ncut
    eigen_val, eigen_vec = ncut.ncut( Adj.numpy(), R )
    # Discretize to get cluster id
    eigenvec_discrete = ncut.discretisation( eigen_vec )
    res = eigenvec_discrete.dot(np.arange(1, R + 1)) 
    # C = np.array(res-1,dtype=np.int64)
    C = torch.tensor(res-1).long()
    return C


# Laplacian eigenvectors
def compute_LapEig(g, pos_enc_dim): # input g is a DGL graph
    Adj = g.adj().to_dense() # Adjacency matrix
    Dn = ( g.in_degrees()** -0.5 ).diag() # Inverse and sqrt of degree matrix
    Lap = torch.eye(g.number_of_nodes()) - Dn.matmul(Adj).matmul(Dn) # Laplacian operator
    EigVal, EigVec = torch.linalg.eig(Lap) # Compute full EVD
    EigVal, EigVec = EigVal.real, EigVec.real # make eig real
    EigVec = EigVec[:, EigVal.argsort()] # sort in increasing order of eigenvalues
    EigVec = EigVec[:,1:pos_enc_dim+1] # select the first non-trivial "pos_enc_dim" eigenvector
    return EigVec



# Plot graphlet orbits
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ORBIT_TEMPLATES = {

    # --------------------------------------------------
    # 2-node graphlet
    # --------------------------------------------------

    0: {
        "edges": [(0, 1)],
        "root": 0,
        "pos": {
            0: (-0.6, 0),
            1: ( 0.6, 0),
        },
    },


    # --------------------------------------------------
    # 3-node graphlets
    # --------------------------------------------------

    # P3: endpoint
    1: {
        "edges": [(0, 1), (1, 2)],
        "root": 0,
        "pos": {
            0: (-0.8, 0),
            1: ( 0.0, 0),
            2: ( 0.8, 0),
        },
    },

    # P3: center
    2: {
        "edges": [(0, 1), (1, 2)],
        "root": 1,
        "pos": {
            0: (-0.8, 0),
            1: ( 0.0, 0),
            2: ( 0.8, 0),
        },
    },

    # Triangle
    3: {
        "edges": [(0, 1), (1, 2), (2, 0)],
        "root": 0,
        "pos": {
            0: ( 0.0,  0.7),
            1: (-0.7, -0.5),
            2: ( 0.7, -0.5),
        },
    },


    # --------------------------------------------------
    # 4-node graphlets
    # --------------------------------------------------

    # P4: endpoint
    4: {
        "edges": [(0, 1), (1, 2), (2, 3)],
        "root": 0,
        "pos": {
            0: (-1.0, 0),
            1: (-0.35, 0),
            2: ( 0.35, 0),
            3: ( 1.0, 0),
        },
    },

    # P4: internal
    5: {
        "edges": [(0, 1), (1, 2), (2, 3)],
        "root": 1,
        "pos": {
            0: (-1.0, 0),
            1: (-0.35, 0),
            2: ( 0.35, 0),
            3: ( 1.0, 0),
        },
    },

    # 3-star: leaf
    6: {
        "edges": [(0, 1), (0, 2), (0, 3)],
        "root": 1,
        "pos": {
            0: ( 0.0,  0.0),
            1: (-0.8, -0.6),
            2: ( 0.8, -0.6),
            3: ( 0.0,  0.9),
        },
    },

    # 3-star: center
    7: {
        "edges": [(0, 1), (0, 2), (0, 3)],
        "root": 0,
        "pos": {
            0: ( 0.0,  0.0),
            1: (-0.8, -0.6),
            2: ( 0.8, -0.6),
            3: ( 0.0,  0.9),
        },
    },

    # 4-cycle
    8: {
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "root": 0,
        "pos": {
            0: (-0.7,  0.7),
            1: ( 0.7,  0.7),
            2: ( 0.7, -0.7),
            3: (-0.7, -0.7),
        },
    },

    # --------------------------------------------------
    # Tailed triangle
    #
    # triangle = 0-1-2-0
    # tail = node 3 connected to node 0
    # --------------------------------------------------

    # Tail
    9: {
        "edges": [
            (0, 1), (1, 2), (2, 0),
            (0, 3),
        ],
        "root": 3,
        "pos": {
            0: ( 0.0,  0.2),
            1: (-0.7, -0.6),
            2: ( 0.7, -0.6),
            3: ( 0.0,  1.0),
        },
    },

    # Degree-2 triangle node
    10: {
        "edges": [
            (0, 1), (1, 2), (2, 0),
            (0, 3),
        ],
        "root": 1,
        "pos": {
            0: ( 0.0,  0.2),
            1: (-0.7, -0.6),
            2: ( 0.7, -0.6),
            3: ( 0.0,  1.0),
        },
    },

    # Attachment node
    11: {
        "edges": [
            (0, 1), (1, 2), (2, 0),
            (0, 3),
        ],
        "root": 0,
        "pos": {
            0: ( 0.0,  0.2),
            1: (-0.7, -0.6),
            2: ( 0.7, -0.6),
            3: ( 0.0,  1.0),
        },
    },


    # --------------------------------------------------
    # Diamond = K4 minus one edge
    #
    # nodes 0 and 1 have degree 3
    # nodes 2 and 3 have degree 2
    # --------------------------------------------------

    # Degree-2 node
    12: {
        "edges": [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
        ],
        "root": 2,
        "pos": {
            0: (-0.65, 0),
            1: ( 0.65, 0),
            2: ( 0.0,  0.75),
            3: ( 0.0, -0.75),
        },
    },

    # Degree-3 node
    13: {
        "edges": [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
        ],
        "root": 0,
        "pos": {
            0: (-0.65, 0),
            1: ( 0.65, 0),
            2: ( 0.0,  0.75),
            3: ( 0.0, -0.75),
        },
    },


    # --------------------------------------------------
    # K4
    # --------------------------------------------------

    14: {
        "edges": [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ],
        "root": 0,
        "pos": {
            0: (-0.65,  0.65),
            1: ( 0.65,  0.65),
            2: ( 0.65, -0.65),
            3: (-0.65, -0.65),
        },
    },
}

def plot_all_orbits(ORBIT_NAMES):
    
    n_rows = 3
    n_cols = 5

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=ORBIT_NAMES,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    for orbit_id in range(15):

        row = orbit_id // n_cols + 1
        col = orbit_id % n_cols + 1

        template = ORBIT_TEMPLATES[orbit_id]

        edges = template["edges"]
        root = template["root"]
        pos = template["pos"]

        # ------------------------
        # edges
        # ------------------------

        edge_x = []
        edge_y = []

        for u, v in edges:

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    color="black",
                    width=2,
                ),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # ------------------------
        # nodes
        # ------------------------

        node_ids = sorted(pos.keys())

        node_x = [pos[v][0] for v in node_ids]
        node_y = [pos[v][1] for v in node_ids]

        colors = [
            "black" if v == root else "white"
            for v in node_ids
        ]

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=18,
                    color=colors,
                    line=dict(
                        color="black",
                        width=2,
                    ),
                ),
                hovertext=[
                    (
                        f"node {v}<br>"
                        + ("root / orbit position"
                           if v == root
                           else "other node")
                    )
                    for v in node_ids
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # remove axes
        fig.update_xaxes(
            visible=False,
            range=[-1.3, 1.3],
            row=row,
            col=col,
        )

        fig.update_yaxes(
            visible=False,
            range=[-1.2, 1.2],
            scaleanchor=f"x{orbit_id + 1}"
            if orbit_id > 0 else "x",
            row=row,
            col=col,
        )

    fig.update_layout(
        title=(
            "Graphlet automorphism orbits 0–14"
            "<br>"
            "<sup>Black node = orbit position being counted</sup>"
        ),
        width=1100,
        height=700,
        template="plotly_white",
        margin=dict(
            l=20,
            r=20,
            t=100,
            b=20,
        ),
    )

    fig.show()







