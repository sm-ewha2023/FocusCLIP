# References
# https://github.com/xk-huang/OrdinalCLIP

import os.path as osp

import torch
import torch.nn as nn
import torchvision.models as models
from CLIP.clip import clip

from focusclip.utils import get_logger

from . import image_encoders
from .builder import MODELS
from .prompt_leaners import PROMPT_LEARNERS
from .prompt_leaners.plain_prompt_learner import PlainPromptLearner
# from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import lib.canny as canny_filter
import torch.nn.functional as F
from CLIP.clip.model import AttentionPool2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(__name__)


@MODELS.register_module()
class FocusCLIP(nn.Module):
    def __init__(
        self,
        text_encoder_name,
        image_encoder_name,
        prompt_learner_cfg,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            logger.info(f"irrelevant kwargs: {kwargs}")

        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )

        # convert to float32
        clip_model.float()
        logger.info("convert `clip_model` to float32. if need fp16 model, call `clip.model.convert_weights`")

        clip_model_edge = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        clip_model_edge.float()
        self.image_encoder_edge = clip_model_edge.visual

        embed_dim = 64 * 32
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.attnpool = AttentionPool2d(clip_model.visual.input_resolution // 32, embed_dim, embed_dim // 64, embed_dim // 2)
        self.norm = nn.LayerNorm(embed_dim)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        prompt_learner_cfg.update(dict(clip_model=clip_model))
        self.prompt_learner: PlainPromptLearner = PROMPT_LEARNERS.build(prompt_learner_cfg)
        self.logit_scale = clip_model.logit_scale

        self.embed_dims = clip_model.text_projection.shape[1]
        self.num_ranks = self.prompt_learner.num_ranks

    def forward(self, images, stains):
        image_features, attn_image_features = self.image_encoder(images)

        canny = canny_filter.Canny(low_threshold=0.1, high_threshold=0.3, hysteresis=False, kernel_size=(3, 3))
        magnitude, edge = canny(images)

        images_edge = edge.repeat(1, 3, 1, 1).data
        image_features_edge, attn_image_features_edge = self.image_encoder_edge(images_edge)

        image_features_norm = self.norm(image_features)
        image_features_edge_norm = self.norm(image_features_edge)

        cross_attn, _ = F.multi_head_attention_forward(
            query=image_features_norm, key=image_features_edge_norm, value=image_features_edge_norm,
            embed_dim_to_check=image_features_norm.shape[-1],
            num_heads=self.attnpool.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        cross_attn_image_features = image_features + cross_attn
        cross_attn_image_features = self.attnpool(cross_attn_image_features)

        cross_attn_image_features = cross_attn_image_features / cross_attn_image_features.norm(dim=-1, keepdim=True)

        attn_image_features = attn_image_features / attn_image_features.norm(dim=-1, keepdim=True)
        attn_image_features_edge = attn_image_features_edge / attn_image_features_edge.norm(dim=-1, keepdim=True)
        sentence_embeds, psudo_sentence_tokens = self.prompt_learner(attn_image_features, attn_image_features_edge, stains)

        logit_scale = self.logit_scale.exp()

        logits = []

        for pts_i, imf_i, tp_i in zip(sentence_embeds, cross_attn_image_features, psudo_sentence_tokens):
            text_features = self.text_encoder(pts_i, tp_i)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits, image_features, text_features

    def forward_text_only(self):
        sentence_embeds = self.prompt_learner()
        psudo_sentence_tokens = self.prompt_learner.psudo_sentence_tokens
        text_features = self.text_encoder(sentence_embeds, psudo_sentence_tokens)

        return text_features

    def encode_image(self, x):
        return self.image_encoder(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype


def load_clip_to_cpu(
    text_encoder_name,
    image_encoder_name,
    root=osp.join(osp.expanduser("~/.cache/clip")),
):
    # text backbone
    if logger is not None:
        print_func = logger.info
    else:
        print_func = print

    print_func("Building CLIP model...")
    text_backbone_name = text_encoder_name
    print_func(f"Text backbone : {text_backbone_name}'s counterpart.")
    url = clip._MODELS[text_backbone_name]
    model_path = clip._download(url, root=root)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    # image backbone
    embed_dim = model.text_projection.shape[1]
    input_resolution = model.visual.input_resolution
    image_backbone_name = image_encoder_name
    print_func(f"Image backbone: {image_backbone_name}")

    if image_backbone_name != text_backbone_name:
        # remove the stochastic back-prop in vgg and alexnet
        MODEL = getattr(image_encoders, image_backbone_name, None)
        if MODEL is None:
            MODEL = getattr(models, image_backbone_name, None)
            logger.warning(f"Try PyTorch Official image model: {image_backbone_name}")
        else:
            logger.info(f"Try Custom image model: {image_backbone_name}")
        if MODEL is None:
            raise ValueError(f"Invalid torchvison model name: {image_backbone_name}")
        model.visual = MODEL(num_classes=embed_dim)
        model.visual.input_resolution = input_resolution
    else:
        print_func(f"CLIP Image encoder: {image_backbone_name}!")

    return model