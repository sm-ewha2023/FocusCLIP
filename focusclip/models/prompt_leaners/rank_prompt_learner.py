# References
# https://github.com/xk-huang/OrdinalCLIP

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from CLIP.clip import clip
from CLIP.clip.model import CLIP

from focusclip.utils import get_logger

from .builder import PROMPT_LEARNERS
from .plain_prompt_learner import PlainPromptLearner

from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(__name__)


@PROMPT_LEARNERS.register_module()
class RankPromptLearner(PlainPromptLearner):
    interpolation_functions = {
        "linear": lambda weights, num_ranks: 1.0 - weights / (num_ranks - 1),
        "inv_prop": lambda weights, _, eps=1e-5: 1.0 / (weights + eps),
        "normal": lambda weights, _: torch.exp(-weights * weights),
    }

    def __init__(
        self,
        clip_model: CLIP,
        num_base_ranks: int,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        stain_tokens_position: str = "tail",
        init_rank_path: Optional[str] = None,
        init_context: Optional[str] = None,
        rank_specific_context: bool = False,
        interpolation_type: str = "linear",
        **kwargs,
    ) -> None:
        super(PlainPromptLearner, self).__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        dtype = dtype = clip_model.token_embedding.weight.dtype

        # context embeds
        context_embeds, _num_context_tokens = self.create_context_embeds(
            clip_model, num_ranks, num_context_tokens, init_context, rank_specific_context, logger, dtype
        )
        num_context_tokens = _num_context_tokens
        self.context_embeds = nn.Parameter(
            context_embeds
        )  # (num_context_tokens, embeds_dim) or (num_ranks, num_context_tokens, embeds_dim)

        # rank embeds
        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * num_base_ranks
        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            clip_model, num_base_ranks, num_tokens_per_rank, init_rank_path, logger, dtype, num_context_tokens
        )
        num_tokens_per_rank = [np.max(_num_tokens_per_rank)] * num_ranks
        self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        assert (
            len(rank_embeds) == num_base_ranks
        ), f"len(rank_embeds) {len(rank_embeds)} == num_base_ranks {num_base_ranks}"

        vis_dim = clip_model.visual.output_dim
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.condition_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.condition_net_edge = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        self.dtype = dtype
        if stain_tokens_position not in self.stain_tokens_position_candidates:
            raise ValueError(f"Invalid rank_tokens_position: {stain_tokens_position}")
        self.stain_tokens_position = stain_tokens_position
        self.clip_model = clip_model

        self.num_context_tokens = num_context_tokens
        self.num_tokens_per_rank = num_tokens_per_rank
        if rank_tokens_position not in self.rank_tokens_position_candidates:
            raise ValueError(f"Invalid rank_tokens_position: {rank_tokens_position}")
        self.rank_tokens_positon = rank_tokens_position
        self.num_ranks = num_ranks
        self.embeddings_dim = clip_model.token_embedding.embedding_dim

        self.create_interpolation_weights(num_base_ranks, num_ranks, interpolation_type, dtype)
        self.num_base_ranks = num_base_ranks

    def create_interpolation_weights(self, num_base_ranks, num_ranks, interpolation_type, dtype):
        if interpolation_type not in self.interpolation_functions:
            raise ValueError(f"Invalide interpolation_type: {interpolation_type}")
        interpolation_func = self.interpolation_functions[interpolation_type]

        interpolation_weights = torch.arange(num_ranks)[..., None].repeat(1, num_base_ranks).to(dtype)
        if num_base_ranks == 1:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, 3)[1:2].to(dtype)
        else:
            base_interpolation_weights = torch.linspace(0, num_ranks - 1, num_base_ranks).to(dtype)
        interpolation_weights = torch.abs(interpolation_weights - base_interpolation_weights[None])
        interpolation_weights = interpolation_func(interpolation_weights, num_ranks)
        interpolation_weights = interpolation_weights / interpolation_weights.sum(dim=-1, keepdim=True)
        self.register_buffer("interpolation_weights", interpolation_weights, persistent=False)

    def forward(self, im_features, im_features_edge, stains):
        rank_embeds = torch.sum(self.interpolation_weights[..., None, None] * self.rank_embeds[None, ...], dim=1)

        context_embeds = self.context_embeds  # (n_ctx, ctx_dim)  (10, 512)
        ctx = context_embeds.unsqueeze(0)  # (1, n_ctx, ctx_dim) (1, 10, 512)
        ctx_gray = ctx[:, :self.num_context_tokens // 2, :]  # (1, n_ctx//2, ctx_dim) (1, 5, 512)
        ctx_edge = ctx[:, self.num_context_tokens // 2:, :]  # (1, n_ctx//2, ctx_dim) (1, 5, 512)

        bias_gray = self.condition_net(im_features)  # (batch, ctx_dim) (32, 512)
        bias_gray = bias_gray.unsqueeze(1)  # (batch, 1, ctx_dim) (32, 1, 512)
        ctx_gray_shifted = ctx_gray + bias_gray  # (batch, n_ctx, ctx_dim) (32, 5, 512)

        bias_edge = self.condition_net_edge(im_features_edge)  # (batch, ctx_dim) (32, 512)
        bias_edge = bias_edge.unsqueeze(1)  # (batch, 1, ctx_dim) (32, 1, 512)
        ctx_edge_shifted = ctx_edge + bias_edge  # (batch, n_ctx, ctx_dim) (32, 5, 512)

        sentence_embeds_batch = []
        psudo_sentence_tokens_batch = []

        for bat_i in range(len(im_features)):
            ctx_gray_i = ctx_gray_shifted[bat_i].unsqueeze(0).expand(self.num_ranks, -1, -1)  # (10, 5, 512)
            ctx_edge_i = ctx_edge_shifted[bat_i].unsqueeze(0).expand(self.num_ranks, -1, -1)  # (10, 5, 512)

            stain_embeds, num_stain_tokens = self.create_stain_embeds(self.clip_model, stains[bat_i],
                                                                      self.dtype)  # (num_stain_token, 512)
            stain_embeds = nn.Parameter(stain_embeds)
            stain_i = stain_embeds.unsqueeze(0).expand(self.num_ranks, -1, -1)  # (10, num_stain_token, 512)


            psudo_sentence_tokens = self.create_psudo_sentence_tokens(
                self.num_tokens_per_rank, self.num_context_tokens, num_stain_tokens, self.num_ranks
            )  # (num_ranks, clip_max_num_tokens)

            sentence_embeds = self.create_sentence_embeds_template(self.clip_model, self.num_ranks, psudo_sentence_tokens.to(device))  # (num_ranks, num_context_tokens, ctx_dim)

            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + num_stain_tokens + _num_tokens_per_rank

                if self.rank_tokens_positon == "tail":
                    if self.stain_tokens_position == 'tail':
                        stain_sentence_embeds = [ctx_gray_i[i], stain_i[i], ctx_edge_i[i]]
                    elif self.stain_tokens_position == 'front':
                        stain_sentence_embeds = [stain_i[i], ctx_gray_i[i], ctx_edge_i[i]]
                    sentence_embeds[i, 1: 1 + pure_sentence_length] = torch.cat(
                        [*stain_sentence_embeds, rank_embeds[i, :_num_tokens_per_rank].to(device)], dim=0)

                elif self.rank_tokens_positon == "front":
                    if self.stain_tokens_position == 'tail':
                        stain_sentence_embeds = [ctx_edge_i[i], ctx_gray_i[i], stain_i[i]]
                    elif self.stain_tokens_position == 'front':
                        stain_sentence_embeds = [ctx_edge_i[i], stain_i[i], ctx_gray_i[i]]
                    sentence_embeds[i, 1: 1 + pure_sentence_length] = torch.cat(
                        [rank_embeds[i, :_num_tokens_per_rank].to(device), *stain_sentence_embeds], dim=0)

                elif self.rank_tokens_positon == "middle":
                    if self.stain_tokens_position == 'tail':
                        stain_sentence_embeds = [ctx_edge_i[i], rank_embeds[i, :_num_tokens_per_rank].to(device),
                                                 ctx_gray_i[i], stain_i[i]]
                    elif self.stain_tokens_position == 'front':
                        stain_sentence_embeds = [ctx_edge_i[i], rank_embeds[i, :_num_tokens_per_rank].to(device),
                                                 stain_i[i], ctx_gray_i[i]]
                    sentence_embeds[i, 1: 1 + pure_sentence_length] = torch.cat(stain_sentence_embeds, dim=0)

            sentence_embeds_batch.append(sentence_embeds)
            psudo_sentence_tokens_batch.append(psudo_sentence_tokens)  # (num_ranks, clip_max_num_tokens)
        sentence_embeds_batch = torch.stack(sentence_embeds_batch)  # (batch, num_ranks, n_ctx, ctx_dim)
        psudo_sentence_tokens_batch = torch.stack(psudo_sentence_tokens_batch)  # (batch, num_ranks, n_ctx)

        return sentence_embeds_batch, psudo_sentence_tokens_batch
