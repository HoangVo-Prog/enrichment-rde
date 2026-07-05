import torch
import torch.nn as nn


class _FusionMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, query, context):
        return self.net(torch.cat([query, context, query * context], dim=-1))


def _two_layer_mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class _RankPartMixerBlock(nn.Module):
    def __init__(self, num_ranks, num_slots, mixer_dim, hidden_part, hidden_rank, hidden_channel):
        super().__init__()
        self.part_norm = nn.LayerNorm(mixer_dim)
        self.part_mlp = _two_layer_mlp(num_slots, hidden_part, num_slots)
        self.rank_norm = nn.LayerNorm(mixer_dim)
        self.rank_mlp = _two_layer_mlp(num_ranks, hidden_rank, num_ranks)
        self.channel_norm = nn.LayerNorm(mixer_dim)
        self.channel_mlp = _two_layer_mlp(mixer_dim, hidden_channel, mixer_dim)

    def forward(self, x, rank_mask=None):
        part_delta = self.part_mlp(self.part_norm(x).permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = x + part_delta
        if rank_mask is not None:
            x = x * rank_mask

        rank_delta = self.rank_mlp(self.rank_norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + rank_delta
        if rank_mask is not None:
            x = x * rank_mask

        x = x + self.channel_mlp(self.channel_norm(x))
        if rank_mask is not None:
            x = x * rank_mask
        return x


class RankPartQueryConditionedMixerAdapter(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_ranks,
        num_slots,
        mixer_dim=256,
        depth=2,
        hidden_part=32,
        hidden_rank=64,
        hidden_channel=512,
        hidden_readout=128,
        context_pooling="mlp",
    ):
        super().__init__()
        if num_ranks < 1:
            raise ValueError("num_ranks must be a positive integer")
        if num_slots < 1:
            raise ValueError("num_slots must be a positive integer")
        if mixer_dim < 1:
            raise ValueError("mixer_dim must be a positive integer")
        if depth < 1:
            raise ValueError("mixer_depth must be a positive integer")
        if context_pooling != "mlp":
            raise ValueError("context_pooling must be mlp")

        self.embed_dim = embed_dim
        self.num_ranks = num_ranks
        self.num_slots = num_slots
        self.mixer_dim = mixer_dim
        self.num_tokens = num_ranks * num_slots
        self.context_pooling = context_pooling

        self.w_in = nn.Linear(embed_dim, mixer_dim)
        self.rank_emb = nn.Parameter(torch.zeros(1, num_ranks, 1, mixer_dim))
        self.part_emb = nn.Parameter(torch.zeros(1, 1, num_slots, mixer_dim))
        self.w_q = nn.Linear(embed_dim, mixer_dim)
        self.film_mlp = _two_layer_mlp(mixer_dim, mixer_dim, mixer_dim * 2)
        self.film_ln = nn.LayerNorm(mixer_dim)
        self.blocks = nn.ModuleList([
            _RankPartMixerBlock(
                num_ranks=num_ranks,
                num_slots=num_slots,
                mixer_dim=mixer_dim,
                hidden_part=hidden_part,
                hidden_rank=hidden_rank,
                hidden_channel=hidden_channel,
            )
            for _ in range(depth)
        ])
        self.final_ln = nn.LayerNorm(mixer_dim)
        self.readout_mlp = _two_layer_mlp(self.num_tokens, hidden_readout, 1)
        self.w_out = nn.Linear(mixer_dim, embed_dim)
        self.last_diagnostics = {}

        nn.init.trunc_normal_(self.rank_emb, std=0.02)
        nn.init.trunc_normal_(self.part_emb, std=0.02)

    def _pad_to_configured_ranks(self, prototype_bank):
        batch_size, num_ranks, num_slots, dim = prototype_bank.shape
        if dim != self.embed_dim:
            raise ValueError(
                "B_q_M last dimension must match adapter embed_dim: "
                f"got {dim}, expected {self.embed_dim}"
            )
        if num_slots != self.num_slots:
            raise ValueError(
                "B_q_M prototype-slot dimension must match adapter num_slots: "
                f"got {num_slots}, expected {self.num_slots}"
            )
        if num_ranks > self.num_ranks:
            raise ValueError(
                "B_q_M rank dimension exceeds adapter num_ranks: "
                f"got {num_ranks}, expected at most {self.num_ranks}"
            )
        if num_ranks == self.num_ranks:
            return prototype_bank, None

        pad = prototype_bank.new_zeros(
            batch_size,
            self.num_ranks - num_ranks,
            num_slots,
            dim,
        )
        padded = torch.cat([prototype_bank, pad], dim=1)
        rank_mask = prototype_bank.new_zeros(1, self.num_ranks, 1, 1)
        rank_mask[:, :num_ranks] = 1.0
        return padded, rank_mask

    def _weight_norm(self, modules):
        total = None
        for module in modules:
            for parameter in module.parameters():
                if parameter.dim() < 2:
                    continue
                value = parameter.detach().float().norm()
                total = value if total is None else total + value
        if total is None:
            return self.rank_emb.detach().float().sum() * 0.0
        return total

    def _weight_diagnostics(self):
        return {
            "mixer/rank_mixing_weight_norm": self._weight_norm([block.rank_mlp for block in self.blocks]),
            "mixer/part_mixing_weight_norm": self._weight_norm([block.part_mlp for block in self.blocks]),
            "mixer/channel_mixing_weight_norm": self._weight_norm([block.channel_mlp for block in self.blocks]),
            "mixer/readout_weight_norm": self._weight_norm([self.readout_mlp]),
        }

    def _mlp_pool(self, h_flat):
        h_t = h_flat.transpose(1, 2)
        return self.readout_mlp(h_t).squeeze(-1)

    def forward(self, z_q, B_q_M):
        if z_q.dim() != 2:
            raise ValueError(f"z_q must have shape [B, d], got {tuple(z_q.shape)}")
        if B_q_M.dim() != 4:
            raise ValueError(f"B_q_M must have shape [B, M, P+1, d], got {tuple(B_q_M.shape)}")
        if z_q.shape[0] != B_q_M.shape[0]:
            raise ValueError("z_q and B_q_M batch dimensions must match")
        if z_q.shape[1] != self.embed_dim:
            raise ValueError(
                "z_q last dimension must match adapter embed_dim: "
                f"got {z_q.shape[1]}, expected {self.embed_dim}"
            )

        B_q_M, rank_mask = self._pad_to_configured_ranks(B_q_M.float())
        rank_mask = None if rank_mask is None else rank_mask.to(device=B_q_M.device, dtype=B_q_M.dtype)

        x = self.w_in(B_q_M)
        x = x + self.rank_emb.to(dtype=x.dtype) + self.part_emb.to(dtype=x.dtype)
        if rank_mask is not None:
            x = x * rank_mask

        q = self.w_q(z_q.float())
        scale, shift = self.film_mlp(q).chunk(2, dim=-1)
        scale = torch.tanh(scale)
        x = self.film_ln(x) * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        if rank_mask is not None:
            x = x * rank_mask

        for block in self.blocks:
            x = block(x, rank_mask=rank_mask)

        h_structured = self.final_ln(x)
        if rank_mask is not None:
            h_structured = h_structured * rank_mask
        h_flat = h_structured.flatten(1, 2)
        h = self._mlp_pool(h_flat)
        c_q = self.w_out(h)

        with torch.no_grad():
            diagnostics = {
                "mixer/context_norm": c_q.detach().float().norm(dim=1).mean(),
                "mixer/film_scale_mean": scale.detach().float().mean(),
                "mixer/film_scale_std": scale.detach().float().std(unbiased=False),
                "mixer/film_shift_mean": shift.detach().float().mean(),
                "mixer/film_shift_std": shift.detach().float().std(unbiased=False),
                "mixer/H_mean": h_structured.detach().float().mean(),
                "mixer/H_std": h_structured.detach().float().std(unbiased=False),
                "mixer/H_flat_token_std": h_flat.detach().float().std(dim=1, unbiased=False).mean(),
                "mixer/readout_output_norm": h.detach().float().norm(dim=1).mean(),
            }
            diagnostics.update(self._weight_diagnostics())
            self.last_diagnostics = diagnostics

        return c_q
