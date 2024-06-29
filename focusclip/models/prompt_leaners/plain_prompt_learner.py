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

from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(__name__)


@PROMPT_LEARNERS.register_module()
class PlainPromptLearner(nn.Module):
    clip_max_num_tokens = 77  # CLIP num_context_tokens = 77
    rank_tokens_position_candidates = {"tail", "middle", "front"}
    stain_tokens_position_candidates = {"tail", "middle", "front"}

    def __init__(
        self,
        clip_model: CLIP,
        num_ranks: int,
        num_tokens_per_rank: Union[int, List],
        num_context_tokens: int,
        rank_tokens_position: str = "tail",
        stain_tokens_position: str = "tail",
        init_rank_path: Optional[str] = None,
        init_context: Optional[str] = None,
        rank_specific_context: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        dtype = clip_model.token_embedding.weight.dtype

        # context embeds of gray
        context_embeds, _num_context_tokens = self.create_context_embeds(
            clip_model, num_ranks, num_context_tokens, init_context, rank_specific_context, logger, dtype
        )
        num_context_tokens = _num_context_tokens
        self.context_embeds = nn.Parameter(
            context_embeds
        )  # (num_context_tokens, embeds_dim) or (num_ranks, num_context_tokens, embeds_dim)

        # rank embeds
        if isinstance(num_tokens_per_rank, int):
            num_tokens_per_rank = [num_tokens_per_rank] * num_ranks
        rank_embeds, _num_tokens_per_rank = self.create_rank_embeds(
            clip_model, num_ranks, num_tokens_per_rank, init_rank_path, logger, dtype, num_context_tokens
        )   # rank_embeds = (10,1,512), _num_tokens_per_rank = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        num_tokens_per_rank = _num_tokens_per_rank
        self.rank_embeds = nn.Parameter(rank_embeds)  # (num_ranks, max_num_tokens_per_rank, embeddings_dim)
        assert len(rank_embeds) == num_ranks, f"len(rank_embeds) {len(rank_embeds)} == num_ranks {num_ranks}"

        # #<use condition-net>
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


    def forward(self, im_features, im_features_edge, stains):
        context_embeds = self.context_embeds  # (n_ctx, ctx_dim)  (10, 512)
        ctx = context_embeds.unsqueeze(0)  # (1, n_ctx, ctx_dim) (1, 10, 512)
        ctx_gray = ctx[:, :self.num_context_tokens // 2, :]  # (1, n_ctx//2, ctx_dim) (1, 5, 512)
        ctx_edge = ctx[:, self.num_context_tokens // 2:, :]  # (1, n_ctx//2, ctx_dim) (1, 5, 512)

        bias_gray = self.condition_net(im_features)  # (batch, ctx_dim) (32, 512)
        bias_gray = bias_gray.unsqueeze(1)  # (batch, 1, ctx_dim) (32, 1, 512)
        ctx_gray_shifted = ctx_gray + bias_gray  # (batch, n_ctx, ctx_dim) (32, 5, 512)

        # # bias of edge
        bias_edge = self.condition_net_edge(im_features_edge)  # (batch, ctx_dim) (32, 512)
        bias_edge = bias_edge.unsqueeze(1)  # (batch, 1, ctx_dim) (32, 1, 512)
        ctx_edge_shifted = ctx_edge + bias_edge  # (batch, n_ctx, ctx_dim) (32, 5, 512)

        sentence_embeds_batch = []
        psudo_sentence_tokens_batch = []

        for bat_i in range(len(im_features)):  # range(batch)
            ctx_gray_i = ctx_gray_shifted[bat_i].unsqueeze(0).expand(self.num_ranks, -1, -1)  # (10, 5, 512)
            ctx_edge_i = ctx_edge_shifted[bat_i].unsqueeze(0).expand(self.num_ranks, -1, -1)  # (10, 5, 512)

            stain_embeds, num_stain_tokens = self.create_stain_embeds(self.clip_model, stains[bat_i], self.dtype)  # (num_stain_token, 512)
            stain_embeds = nn.Parameter(stain_embeds)
            stain_i = stain_embeds.unsqueeze(0).expand(self.num_ranks, -1, -1)  # (10, num_stain_token, 512)

            psudo_sentence_tokens = self.create_psudo_sentence_tokens(
                self.num_tokens_per_rank, self.num_context_tokens, num_stain_tokens, self.num_ranks
            )  # (num_ranks, clip_max_num_tokens)

            sentence_embeds = self.create_sentence_embeds_template(self.clip_model, self.num_ranks, psudo_sentence_tokens.to(device))  # (num_ranks, num_context_tokens, ctx_dim)

            for i in range(self.num_ranks):
                _num_tokens_per_rank = self.num_tokens_per_rank[i]
                pure_sentence_length = self.num_context_tokens + num_stain_tokens + _num_tokens_per_rank

                stain_sentence_embeds = []
                if self.rank_tokens_positon == "tail":
                    if self.stain_tokens_position == 'tail':
                        stain_sentence_embeds = [ctx_gray_i[i], stain_i[i], ctx_edge_i[i]]
                    elif self.stain_tokens_position == 'front':
                        stain_sentence_embeds = [stain_i[i], ctx_gray_i[i], ctx_edge_i[i]]
                    sentence_embeds[i, 1: 1 + pure_sentence_length] = torch.cat([*stain_sentence_embeds, self.rank_embeds[i, :_num_tokens_per_rank].to(device)], dim=0)

                elif self.rank_tokens_positon == "front":
                    if self.stain_tokens_position == 'tail':
                        stain_sentence_embeds = [ctx_edge_i[i], ctx_gray_i[i], stain_i[i]]
                    elif self.stain_tokens_position == 'front':
                        stain_sentence_embeds = [ctx_edge_i[i], stain_i[i], ctx_gray_i[i]]
                    sentence_embeds[i, 1: 1 + pure_sentence_length] = torch.cat([self.rank_embeds[i, :_num_tokens_per_rank].to(device), *stain_sentence_embeds], dim=0)

                elif self.rank_tokens_positon == "middle":
                    if self.stain_tokens_position == 'tail':
                        stain_sentence_embeds = [ctx_edge_i[i], self.rank_embeds[i, :_num_tokens_per_rank].to(device), ctx_gray_i[i], stain_i[i]]
                    elif self.stain_tokens_position == 'front':
                        stain_sentence_embeds = [ctx_edge_i[i], self.rank_embeds[i, :_num_tokens_per_rank].to(device), stain_i[i], ctx_gray_i[i]]
                    sentence_embeds[i, 1: 1 + pure_sentence_length] = torch.cat(stain_sentence_embeds, dim=0)

            sentence_embeds_batch.append(sentence_embeds)
            psudo_sentence_tokens_batch.append(psudo_sentence_tokens)  # (num_ranks, clip_max_num_tokens)
        sentence_embeds_batch = torch.stack(sentence_embeds_batch)  # (batch, num_ranks, n_ctx, ctx_dim)
        psudo_sentence_tokens_batch = torch.stack(psudo_sentence_tokens_batch)  # (batch, num_ranks, n_ctx)

        return sentence_embeds_batch, psudo_sentence_tokens_batch

    def create_sentence_embeds_template(self, clip_model, num_ranks, psudo_sentence_tokens):
        with torch.no_grad():
            null_embed = clip_model.token_embedding(torch.LongTensor([0]).to(device))[0]
            sot_embed = clip_model.token_embedding(torch.LongTensor([49406]).to(device))[0]
            eot_embed = clip_model.token_embedding(torch.LongTensor([49407]).to(device))[0]
            full_stop_embed = clip_model.token_embedding(torch.LongTensor([269]).to(device))[0]

        sentence_embeds = null_embed[None, None].repeat(
            num_ranks, self.clip_max_num_tokens, 1
        )  # not the same null_embed!
        argmax_index = psudo_sentence_tokens.argmax(dim=-1)
        rank_index = torch.arange(num_ranks)

        sentence_embeds[:, 0, :] = sot_embed
        sentence_embeds[rank_index, argmax_index] = eot_embed
        sentence_embeds[rank_index, argmax_index - 1] = full_stop_embed

        return sentence_embeds

    def create_psudo_sentence_tokens(self, num_tokens_per_rank, num_context_tokens, num_stain_tokens, num_ranks):
        psudo_sentence_tokens = torch.zeros(num_ranks, self.clip_max_num_tokens, dtype=torch.long)
        if isinstance(num_tokens_per_rank, List):
            assert num_ranks == len(num_tokens_per_rank)
            for i, _num_tokens_per_rank in enumerate(num_tokens_per_rank):
                # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
                sentence_length = 1 + num_context_tokens + num_stain_tokens + _num_tokens_per_rank + 1 + 1
                psudo_sentence_tokens[i, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)
        else:
            # <sot>, <context_0>, ..., <context_N>, <rank_i>, <full_stop>, <eot>
            sentence_length = 1 + num_context_tokens + num_stain_tokens + num_tokens_per_rank + 1 + 1
            psudo_sentence_tokens[:, :sentence_length] = torch.arange(0, sentence_length, dtype=torch.long)

        return psudo_sentence_tokens.to(device)

    def create_rank_embeds(
        self, clip_model, num_ranks, num_tokens_per_rank, init_rank_path, logger, dtype, num_context_tokens
    ):
        if init_rank_path is not None:
            logger.info(f"load init rank from: {init_rank_path}.")

            rank_names = self.read_rank_file(init_rank_path, logger)
            if len(rank_names) != num_ranks:
                raise ValueError(
                    f"The length of rank_names is {len(rank_names)}, which is not equal to num_ranks {num_ranks}"
                )

            _rank_tokens = [clip._tokenizer.encode(rank_name) for rank_name in rank_names]
            _num_tokens_per_rank = [len(rank_token) for rank_token in _rank_tokens]
            logger.info(f"num_tokens_per_rank: {num_tokens_per_rank} -> {_num_tokens_per_rank}")
            num_tokens_per_rank = _num_tokens_per_rank
            max_num_tokens_per_rank = np.max(num_tokens_per_rank)

            rank_tokens = torch.zeros(len(_rank_tokens), max_num_tokens_per_rank, dtype=torch.long)
            for i, rank_token in enumerate(_rank_tokens):
                # 3 is <eot>, <sot>, and <full_stop>
                valid_length = self.clip_max_num_tokens - num_context_tokens - 3
                if len(rank_token) > valid_length:
                    rank_token = rank_token[:valid_length]
                    raise ValueError(f"rank tokens are too long: {rank_token}")
                rank_tokens[i, : len(rank_token)] = torch.LongTensor(rank_token)
            rank_embeds = clip_model.token_embedding(rank_tokens).type(dtype)
            rank_embeds = rank_embeds[:, :max_num_tokens_per_rank]

        else:
            logger.info(f"num rank: {num_ranks}")
            logger.info(f"num_tokens_per_rank: {num_tokens_per_rank}")
            embeddings_dim = clip_model.token_embedding.embedding_dim
            if isinstance(num_tokens_per_rank, List):
                max_num_tokens_per_rank = np.max(num_tokens_per_rank)
            else:
                max_num_tokens_per_rank = num_tokens_per_rank
            if self.clip_max_num_tokens < num_context_tokens + max_num_tokens_per_rank + 3:
                raise ValueError(f"rank tokens are too long: {rank_token}")
            rank_embeds = torch.empty((num_ranks, max_num_tokens_per_rank, embeddings_dim), dtype=dtype)    # (10,1,512)
            nn.init.normal_(rank_embeds, std=0.02)

        return (rank_embeds, num_tokens_per_rank)

    def read_rank_file(self, init_rank_path, logger):
        rank_names = []
        with open(init_rank_path, "r") as f:
            for line in f.readlines():
                line = line.strip().replace("_", " ")
                rank_names.append(line)
        logger.info(f"num rank: {len(rank_names)}:\n\t{rank_names[:5]}\n\t{rank_names[-5:]}")
        return rank_names

    def create_context_embeds(
        self,
        clip_model,
        num_ranks: int,
        num_context_tokens: int,
        init_context: Optional[str],
        rank_specific_context: bool,
        logger,
        dtype,
    ):
        # context embeddings
        logger.info("init context token")
        if init_context is not None:
            init_context = init_context.replace("_", " ")
            logger.info(f"init context: {init_context}")

            prompt_tokens = clip.tokenize(init_context)
            prompt_tokens = prompt_tokens[0]  # (num_context_tokens=77)
            _num_context_tokens = torch.argmax(prompt_tokens).item() - 1    # _num_context_tokens of 'a photo of a {}.' = 7
            logger.info(f"num_context_tokens: {num_context_tokens} -> {_num_context_tokens}")   # num_context_tokens: 10 -> 7
            num_context_tokens = _num_context_tokens

            with torch.no_grad():
                context_embeds = clip_model.token_embedding(prompt_tokens).type(dtype)  # (77, 512)
            context_embeds = context_embeds[1 : 1 + num_context_tokens]

            logger.info(f"rank_specific_context: {rank_specific_context}")
            if rank_specific_context is True:
                context_embeds = context_embeds[None].repeat(num_ranks, 1, 1)   # (10, 7, 512)
        else:   #random initialization
            embeds_dim = clip_model.token_embedding.embedding_dim   # embeds_dim=512
            init_context = " ".join(["X"] * num_context_tokens) # 'X X X X X X X X X X', num_context_tokens = 10

            logger.info(f"random context: {init_context}")
            logger.info(f"num context tokens: {num_context_tokens}")
            logger.info(f"rank_specific_context: {rank_specific_context}")
            if rank_specific_context is True:
                context_embeds = torch.empty((num_ranks, num_context_tokens, embeds_dim), dtype=dtype)  # context_embeds = (10, 10, 512)
            else:
                context_embeds = torch.empty((num_context_tokens, embeds_dim), dtype=dtype) # context_embeds = (10, 512)
            nn.init.normal_(context_embeds, std=0.02)

        return context_embeds, num_context_tokens

    def create_stain_embeds(
        self,
        clip_model,
        stains,
        dtype,
    ):
        stain_prompt_tokens = clip.tokenize(stains)[0].to(device)  # clip.tokenize(stains) = (1,77)
        num_stain_tokens = torch.argmax(stain_prompt_tokens).item() - 1

        with torch.no_grad():
            stain_embeds = clip_model.token_embedding(stain_prompt_tokens).type(dtype)  # (77, 512)
        stain_embeds = stain_embeds[1: 1 + num_stain_tokens]

        return stain_embeds, num_stain_tokens
