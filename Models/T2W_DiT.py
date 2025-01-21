from .Basics import *

class Block(nn.Module):
    def __init__(self,
                 n_paths: int,
                 d_in: int,
                 d_out: int,
                 d_context: int,
                 d_time: int,
                 n_heads: int = 8,
                 dropout: float = 0.0,
                 ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_context = d_context
        self.n_heads = n_heads
        self.dropout = dropout

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_in),
            Swish()
        )

        d_mid = d_in + d_context
        self.res = nn.Sequential(
            Rearrange("B (N L) D", "(B N) D L", N=n_paths),
            SERes1D(d_mid, d_mid, d_mid),
            SERes1D(d_mid, d_mid, d_mid),
            Rearrange("(B N) D L", "B (N L) D", N=n_paths)
        )

        self.attn = AttentionBlock(
            d_in=d_mid,
            d_head=64,
            d_expand=d_mid * 2,
            d_out=d_in,
            d_time=d_in,
            n_heads=self.n_heads,
            dropout=self.dropout
        )

        self.out_proj = nn.Sequential(
            Swish(),
            nn.Linear(d_in, d_out)
        )

        self.shortcut = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

    def forward(self, x, context, t):
        # x: (B, N, L, D)
        # context: (B, N, L, D')
        residual = self.shortcut(x)
        x = self.res(torch.cat([x, context], dim=-1))   # (B, N, L, D)
        x = self.attn(x, self.time_proj(t))
        return self.out_proj(x) + residual


class T2W_DiT(nn.Module):
    def __init__(self, D_in: int, N_routes: int, L_route: int, L_traj: int, d_context: int, n_layers: int, T: int):
        super().__init__()
        self.D_in = D_in
        self.N_routes = N_routes
        self.L_route = L_route
        self.L_traj = L_traj
        self.d_context = d_context
        self.n_layers = n_layers
        self.T = T

        self.time_embed = nn.Sequential(
            nn.Embedding(T, 128),
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(-1, (1, -1))
        )

        # Input: (B, N, L, 2)

        self.route_proj = nn.Sequential(
            Rearrange("B N L D", "(B N) D L"),
            nn.Conv1d(D_in, 64, 3, 1, 1),
            Swish(),
            nn.Conv1d(64, 256, 3, 1, 1),
            Rearrange("(B N) D L", "B (N L) D", N=N_routes),
        )

        self.traj_proj = nn.Sequential(
            # Trajectory inner feature extraction
            # traj -> feature sequence
            Rearrange("B N L D", "(B N) D L"),  # (BN, 2, L')
            nn.Conv1d(d_context, 128, 3, 2, 1), Swish(),
            *[SERes1D(128, 256, 128) for _ in range(4)],
            nn.Conv1d(128, 256, 3, 1, 1),

            # Attention among all traj tokens
            Rearrange("(B N) D L", "B (N L) D", N=N_routes),
            AttentionBlock(256, 64, 512, 256, 4),
            AttentionBlock(256, 64, 512, 256, 4),
        )

        self.stages = SequentialWithAdditionalInputs(*[
            Block(N_routes, 256, 256, 256, 256, 8, dropout=0.15)
            for _ in range(n_layers)
        ])

        # (B, N, L, 128)
        self.head = nn.Sequential(
            Rearrange("B (N L) D",  "B N L D", N=N_routes),
            nn.Linear(256, 64),
            Swish(),
            nn.Linear(64, D_in)
        )

    def forward(self, x, context, t):
        """
        :param segments: (B, N_segs, C_seg)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
        :return:
        """

        t = self.time_embed(t)
        x = self.route_proj(x)
        context = self.traj_proj(context)

        x = self.stages(x, context, t)

        return self.head(x)
