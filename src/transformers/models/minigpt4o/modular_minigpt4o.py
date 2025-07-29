# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import math
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, SlidingWindowLayer
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ..auto import AutoModel


logger = logging.get_logger(__name__)


class MiniGPT4OTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniGPT4OTextModel`]. It is used to instantiate an
    MiniGPT4OTextModel model according to the specified arguments, defining the model architecture.

    Configuration objects that inherit from [`MiniGPT4OTextConfig`] and can be used to control the model outputs. Read
    the documentation from [`MiniGPT4OTextConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 200000):
            Vocabulary size of the MiniGPT4OText model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MiniGPT4OTextModel`]
        vocab_size_per_layer_input (`int`, *optional*, defaults to 199744):
            Vocabulary size of the per-layer text embeddings that augment the standard embeddings.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        hidden_size_per_layer_input (`int`, *optional*, defaults to 256):
            Dimension of the hidden representations for per-layer embeddings.
        intermediate_size (`int` or `Sequence[int]`, *optional*, defaults to 16384):
            Dimension of the MLP representations. MatFormer configurations may wish to provide a sequence of integers
            to account for variable intermediate_size values across layers. In such cases,
            `len(intermediate_size) == num_hidden_layers`.
        num_hidden_layers (`int`, *optional*, defaults to 35):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings used in global attention.
        rope_local_base_freq (float, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings for local attention.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to 512):
            This is the size of the sliding window used by local attention layers.
        layer_types (`Optional`, *optional*):
            A sequence of strings defining the attention type for that layer as either "sliding_attention" or
            "full_attention". If not provided, `layer_types` will be inferred from `num_hidden_layers` using a pattern
            of four "sliding_attention" layers followed one "full_attention". The last layer in the model should always
            be a "full_attention" layer.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            Scaling factor when applying tanh softcapping on the logits.
        altup_active_idx (`int`, *optional*, defaults to 0):
            The index of the prediction from which AltUp will compute additional predictions or correct
        altup_coef_clip (`float`, *optional*, defaults to 120.0):
            The maximum amplitude of an AltUp prediction or correction coefficient weight.
        altup_correct_scale (`bool`, *optional*, defaults to `True`):
            If True, apply the `AltUp.correct_output_scale` to the corrected prediction at `altup_active_idx`.
        altup_num_inputs (`int`, *optional*, defaults to 4):
            The number of predictions that AltUp should make given the input sequence.
        num_kv_shared_layers (`int`, *optional*, defaults to 15):
            The number of layers that share KV cache values.
        laurel_rank (int, *optional*, defaults to 64):
            The intermediate size for the linear projections in the Learned Augmented Residual Layer.
        activation_sparsity_pattern (Sequence[float], *optional*, defaults to `(0.95,) * 10 + (0.0,) * 25`):
            The sparsity factor used to extract the top-k activations for a given layer. The provided Sequence must
            explicitly provide a sparsity value for each layer in the model.

    ```python
    >>> from transformers import MiniGPT4OTextModel, MiniGPT4OTextConfig

    >>> # Initializing a MiniGPT4OText configuration
    >>> configuration = MiniGPT4OTextConfig()

    >>> # Initializing a model from the configuration
    >>> model = MiniGPT4OTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "minigpt4o_text"

    def __init__(
        self,
        vocab_size: int = 200_000,
        vocab_size_per_layer_input: int = 199_744,
        hidden_size: int = 2048,
        hidden_size_per_layer_input: int = 256,
        intermediate_size: Union[int, Sequence[int]] = 16_384,
        num_hidden_layers: int = 35,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_activation: str = "gelu_pytorch_tanh",
        max_position_embeddings: int = 32_768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        rope_theta: float = 1_000_000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        rope_local_base_freq: float = 10_000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 512,
        layer_types: Optional[Sequence[str]] = None,
        final_logit_softcapping: float = 30.0,
        altup_active_idx: int = 0,
        altup_coef_clip: float = 120.0,
        altup_correct_scale: bool = True,
        altup_num_inputs: int = 4,
        num_kv_shared_layers: int = 15,
        laurel_rank: int = 64,
        activation_sparsity_pattern: Optional[Union[float, Sequence[float]]] = (0.95,) * 10 + (0.0,) * 25,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        if isinstance(intermediate_size, Sequence) and (intsize_len := len(intermediate_size)) != num_hidden_layers:
            raise ValueError(
                "intermediate_size must have an explicit intermediate size for every layer or one for all layers. "
                f"Expected {num_hidden_layers} values but got {intsize_len}."
            )
        elif not isinstance(intermediate_size, Sequence):
            intermediate_size = [intermediate_size] * num_hidden_layers

        self.vocab_size = vocab_size
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.layer_types = layer_types

        self.rope_local_base_freq = rope_local_base_freq
        self.rope_scaling = rope_scaling
        rope_config_validation(self)

        if layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        layer_type_validation(self.layer_types)

        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.num_kv_shared_layers = num_kv_shared_layers

        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.altup_correct_scale = altup_correct_scale
        self.altup_num_inputs = altup_num_inputs

        self.laurel_rank = laurel_rank

        if activation_sparsity_pattern is None:
            activation_sparsity_pattern = [0.0] * num_hidden_layers

        if (len_asp := len(activation_sparsity_pattern)) != num_hidden_layers:
            raise ValueError(
                "activation_sparsity_pattern must have an explicit activation sparsity value for every layer."
                f"Expected {num_hidden_layers} values but got {len_asp}."
            )
        self.activation_sparsity_pattern = activation_sparsity_pattern


class MiniGPT4OAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniGPT4OAudioEncoder`]. It is used to instantiate
    an `MiniGPT4OAudioEncoder` model according to the specified arguments, defining the model architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 128):
            Vocabulary size of the additional hard-token embeddings for audio model.
        vocab_offset (`int`, *optional*, defaults to 200128):
            Offset between the tokenizer vocab index for the token ids embedded by `MiniGPT4OMultimodalEmbedder` and the
            0-indexed `MiniGPT4OMultimodalEmbedder.embedding` table.
        input_feat_size (`int`, *optional*, defaults to 128):
            The number of channels in each mel-spectrogram frame.
        hidden_size (`int`, *optional*, defaults to 1536):
            Dimension of the hidden representations.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        gradient_clipping (`float`, *optional*, defaults to 10000000000.0):
            Clipping value used to stabilize extremely large gradient values.
        conf_attention_chunk_size (`int`, *optional*, defaults to 12):
            The sub-sequence size for local attention processing inside the Conformer section.
        conf_attention_context_left (`int`, *optional*, defaults to 13):
            The left context size of the local attention inside the Conformer section.
        conf_attention_context_right (`int`, *optional*, defaults to 0):
            The right context size of the local attention inside the Conformer section.
        conf_attention_logit_cap (`float`, *optional*, defaults to 50.0):
            Logit cap applied during local attention inside the Conformer section.
        conf_num_attention_heads (`int`, *optional*, defaults to 8):
            The number of attention heads in local attention inside the Conformer section.
        conf_num_hidden_layers (`int`, *optional*, defaults to 12):
            The number of layers that use local attention inside the Conformer section.
        conf_conv_kernel_size (`int`, *optional*, defaults to 5):
            Convolution kernel size for the conformer block inside the Conformer section.
        conf_reduction_factor (`int`, *optional*, defaults to 4):
            Reduction factor used in the conformer block inside the Conformer section.
        conf_residual_weight (`float`, *optional*, defaults to 0.5):
            Residual connection weight inside the Conformer section.
        sscp_conv_channel_size (`tuple(int, int)`, *optional*, defaults to `(128, 32)`):
            The channel sizes for the first and second convolutional layers in the Sub-sample Convolution Projection.
        sscp_conv_group_norm_eps (`float`, *optional*, defaults to 0.001):
            Epsilon used in group normalization in the subsample convolution projection.
        sscp_conv_kernel_size (`tuple(tuple(int, int), tuple(int, int))`, *optional*, defaults to `((3, 3), (3, 3))`):
            Kernel sizes of the two convolutional layers in the subsample convolution projection.
        sscp_conv_stride_size (`tuple(tuple(int, int), tuple(int, int))`, *optional*, defaults to `((2, 2), (2, 2))`):
            Stride sizes of the two convolutional layers in the subsample convolution projection.
    """

    model_type = "minigpt4o_audio"

    def __init__(
        self,
        vocab_size: int = 128,
        vocab_offset: int = 200_000 + 128,  # text vocab size + vision vocab size
        input_feat_size: int = 128,
        hidden_size: int = 1536,
        rms_norm_eps: float = 1e-6,
        gradient_clipping: float = 10_000_000_000.0,
        conf_attention_chunk_size: int = 12,
        conf_attention_context_left: int = 13,
        conf_attention_context_right: int = 0,
        conf_attention_logit_cap: float = 50.0,
        conf_num_attention_heads: int = 8,
        conf_num_hidden_layers: int = 12,
        conf_conv_kernel_size: int = 5,
        conf_reduction_factor: int = 4,
        conf_residual_weight: float = 0.5,
        sscp_conv_channel_size: tuple[int, int] = (128, 32),
        sscp_conv_group_norm_eps: float = 1e-3,
        sscp_conv_kernel_size: tuple[tuple[int, int], tuple[int, int]] = (
            (3, 3),
            (3, 3),
        ),
        sscp_conv_stride_size: tuple[tuple[int, int], tuple[int, int]] = (
            (2, 2),
            (2, 2),
        ),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_feat_size = input_feat_size
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset
        self.gradient_clipping = gradient_clipping
        self.conf_attention_chunk_size = conf_attention_chunk_size
        self.conf_attention_context_left = conf_attention_context_left
        self.conf_attention_context_right = conf_attention_context_right
        self.conf_attention_logit_cap = conf_attention_logit_cap
        self.conf_num_attention_heads = conf_num_attention_heads
        self.conf_num_hidden_layers = conf_num_hidden_layers
        self.conf_conv_kernel_size = conf_conv_kernel_size
        self.conf_reduction_factor = conf_reduction_factor
        self.conf_residual_weight = conf_residual_weight
        self.sscp_conv_channel_size = sscp_conv_channel_size
        self.sscp_conv_group_norm_eps = sscp_conv_group_norm_eps
        self.sscp_conv_kernel_size = sscp_conv_kernel_size
        self.sscp_conv_stride_size = sscp_conv_stride_size


class MiniGPT4OVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration for a vision backbone. It is used to
    instantiate a vision model according to the specified arguments, defining the model architecture.

    Args:
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        do_pooling (`bool`, *optional*, defaults to `False`):
            Whether to do pooling for the last_hidden_state or not.
        architecture (`str`, *optional*, defaults to `"mobilenetv5_300m_enc"`):
            Determines vision architecture.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        vocab_size (`int`, *optional*, defaults to 128):
            Vocabulary size of the additional hard-token embeddings for vision model.
        vocab_offset (`int`, *optional*, defaults to 200000):
            Offset between the tokenizer vocab index for the token ids embedded by `MiniGPT4OMultimodalEmbedder` and the
            0-indexed `MiniGPT4OMultimodalEmbedder.embedding` table.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
    """

    model_type = "minigpt4o_vision"

    def __init__(
        self,
        initializer_range: float = 0.02,
        do_pooling: bool = False,
        architecture: str = "mobilenetv5_300m_enc",
        hidden_size: int = 2048,
        vocab_size: int = 128,
        vocab_offset: int = 200_000,
        rms_norm_eps: float = 1e-06,
        model_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.initializer_range = initializer_range
        self.do_pooling = do_pooling
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.vocab_offset = vocab_offset
        self.rms_norm_eps = rms_norm_eps


class MiniGPT4OConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiniGPT4OForConditionalGeneration`]. It is used to
    instantiate a MiniGPT4OForConditionalGeneration according to the specified arguments, defining the model
    architecture.

    Args:
        text_config (`Union[MiniGPT4OTextConfig, dict]`, *optional*):
            The config object of the text backbone.
        vision_config (`Union[MiniGPT4OVisionConfig, dict]`,  *optional*):
            Custom vision config or dict.
        audio_config (`Union[MiniGPT4OAudioConfig, dict]`,  *optional*):
            Custom audio config or dict.
        audio_soft_tokens_per_image (`int`, *optional*, defaults to 188):
            The number of soft tokens per audio clip.
        vision_soft_tokens_per_image (`int`, *optional*, defaults to 256):
            The number of soft tokens per image.
        boi_token_id (`int`, *optional*, defaults to 199999):
            The begin-of-image token index to wrap the image prompt.
        eoi_token_id (`int`, *optional*, defaults to 200000):
            The end-of-image token index to wrap the image prompt.
        image_token_id (`int`, *optional*, defaults to 200001):
            The image token index to encode the image prompt.
        boa_token_id (`int`, *optional*, defaults to 200000):
            The begin-of-audio token index to wrap the audio prompt.
        eoa_token_id (`int`, *optional*, defaults to 200128):
            The end-of-audio token index to wrap the audio prompt.
        audio_token_id (`int`, *optional*, defaults to 200129):
            The audio token index to encode the audio prompt.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "minigpt4o"
    sub_configs = {
        "text_config": MiniGPT4OTextConfig,
        "vision_config": MiniGPT4OVisionConfig,
        "audio_config": MiniGPT4OAudioConfig,
    }

    def __init__(
        self,
        text_config: Optional[Union[MiniGPT4OTextConfig, dict[str, Any]]] = None,
        vision_config: Optional[Union[MiniGPT4OVisionConfig, dict[str, Any]]] = None,
        audio_config: Optional[Union[MiniGPT4OAudioConfig, dict[str, Any]]] = None,
        audio_soft_tokens_per_image: int = 188,
        vision_soft_tokens_per_image: int = 256,
        boi_token_id: int = 199_999,
        eoi_token_id: int = 200_000,
        image_token_id: int = 200_001,
        boa_token_id: int = 200_000,
        eoa_token_id: int = 200_128,
        audio_token_id: int = 200_129,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(text_config, dict):
            text_config = MiniGPT4OTextConfig(**text_config)
        elif text_config is None:
            text_config = MiniGPT4OTextConfig()
            logger.info("text_config is None. Using default MiniGPT4OTextConfig.")

        if isinstance(vision_config, dict):
            vision_config = MiniGPT4OVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = MiniGPT4OVisionConfig()
            logger.info("vision_config is None. Using default MiniGPT4OVisionConfig.")

        if isinstance(audio_config, dict):
            audio_config = MiniGPT4OAudioConfig(**audio_config)
        elif audio_config is None:
            audio_config = MiniGPT4OAudioConfig()
            logger.info("audio_config is None. Using default MiniGPT4OAudioConfig.")

        self.text_config = text_config
        self.vision_config = vision_config
        self.audio_config = audio_config

        self.audio_soft_tokens_per_image = audio_soft_tokens_per_image
        self.vision_soft_tokens_per_image = vision_soft_tokens_per_image
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range


class MiniGPT4OModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    def __init__(self, *args, image_hidden_states=None, audio_hidden_states=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_hidden_states = image_hidden_states
        self.audio_hidden_states = audio_hidden_states


class MiniGPT4OCausalLMOutputWithPast(BaseModelOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    def __init__(self, *args, loss=None, logits=None, image_hidden_states=None, audio_hidden_states=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss
        self.logits = logits
        self.image_hidden_states = image_hidden_states
        self.audio_hidden_states = audio_hidden_states


class MiniGPT4ORMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def _norm(self, x):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()) * self.weight.float()
        return output.type_as(x)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class MiniGPT4ORotaryEmbedding(nn.Module):
    def __init__(self, config: MiniGPT4OTextConfig):
        super().__init__()
        self.config = config
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device=torch.device("cpu")) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = position_ids.float().to(self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class MiniGPT4OAttention(nn.Module):
    def __init__(self, config: MiniGPT4OTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.sliding_window = config.sliding_window

        # Determine if this is a sliding attention layer
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.q_norm = MiniGPT4ORMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniGPT4ORMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = MiniGPT4ORotaryEmbedding(config)

        first_kv_shared_layer_idx = self.config.num_hidden_layers - self.config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        layer_type = config.layer_types[layer_idx]
        self.kv_shared_layer_index = (
            first_kv_shared_layer_idx - 1 - config.layer_types[first_kv_shared_layer_idx - 1 :: -1].index(layer_type)
            if self.is_kv_shared_layer
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and self.kv_shared_layer_index is not None and past_key_value is not None:
            layer = past_key_value.layers[self.kv_shared_layer_index]
            indices = cache_position.to(layer.keys.device)
            if isinstance(layer, SlidingWindowLayer):
                if cache_position.shape[0] > layer.get_max_cache_shape():
                    indices = slice(0, layer.get_max_cache_shape())
                else:
                    indices = indices.clamp(min=0, max=layer.get_max_cache_shape() - 1)

            key_states = layer.keys[:, :, indices].to(query_states.device)
            value_states = layer.values[:, :, indices].to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = self._eager_attention_forward
        if hasattr(self.config, '_attn_implementation') and self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=1.0,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _eager_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        scaling: float = 1.0,
        sliding_window: Optional[int] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Standard attention implementation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output, attn_weights


class MiniGPT4OMLP(nn.Module):
    def __init__(self, config: MiniGPT4OTextConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(self.activation_sparsity, dtype=torch.float32, device=inputs.device)
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier: torch.Tensor = normal_dist.icdf(target_sparsity_tensor)
        std_multiplier = std_multiplier.type(inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return nn.functional.relu(inputs - cutoff_x)


class MiniGPT4OAltUp(nn.Module):
    """Alternating Updates (AltUp)"""

    def __init__(self, config: MiniGPT4OTextConfig):
        super().__init__()
        self.config = config
        self.correct_output_scale = nn.Parameter(torch.zeros(self.config.hidden_size))
        self.correction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.config.altup_num_inputs, self.config.altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(self.config.hidden_size, self.config.altup_num_inputs, bias=False)
        self.router_norm = MiniGPT4ORMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(self.config.hidden_size**-1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        modalities = self.compute_router_modalities(hidden_states[self.config.altup_active_idx])

        if self.training and self.config.altup_coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        all_coefs: torch.Tensor = (
            self.prediction_coefs(modalities)
            .reshape(*modalities.shape[:-1], self.config.altup_num_inputs, self.config.altup_num_inputs)
            .permute(0, 1, 3, 2)
        )

        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)
        predictions += hidden_states
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]
        innovation = innovation.repeat(self.config.altup_num_inputs, 1, 1, 1)

        if self.config.altup_coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-self.config.altup_coef_clip, self.config.altup_coef_clip)

        all_coefs: torch.Tensor = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        corrected = torch.mul(innovation, all_coefs)
        corrected += predictions
        return corrected.contiguous().type_as(activated)

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        return self.forward(corrected)


class MiniGPT4OLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer"""

    def __init__(self, config: MiniGPT4OTextConfig):
        super().__init__()
        self.config = config

        self.linear_left = nn.Linear(self.config.hidden_size, self.config.laurel_rank, bias=False)
        self.linear_right = nn.Linear(self.config.laurel_rank, self.config.hidden_size, bias=False)
        self.post_laurel_norm = MiniGPT4ORMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states: torch.Tensor = self.linear_left(hidden_states)
        laurel_hidden_states: torch.Tensor = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed_laurel_hidden_states


class MiniGPT4OTextDecoderLayer(nn.Module):
    def __init__(self, config: MiniGPT4OTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = ACT2FN[config.hidden_activation]

        self.altup = MiniGPT4OAltUp(config)
        self.laurel = MiniGPT4OLaurelBlock(config)
        self.self_attn = MiniGPT4OAttention(config, layer_idx)
        self.mlp = MiniGPT4OMLP(config, layer_idx=layer_idx)
        
        self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
        self.post_per_layer_input_norm = MiniGPT4ORMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.input_layernorm = MiniGPT4ORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniGPT4ORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = MiniGPT4ORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = MiniGPT4ORMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        per_layer_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # apply global RoPE to non-sliding layer only
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        attn, self_attn_weights = self.self_attn(
            hidden_states=active_prediction_normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        attn = self.post_attention_layernorm(attn)

        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm
        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.config.altup_active_idx].clone()
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        # per_layer_input_gate
        first_prediction = self.per_layer_input_gate(first_prediction)
        first_prediction = self.act_fn(first_prediction)
        first_prediction = torch.multiply(first_prediction, per_layer_input)

        # per_layer_projection
        first_prediction = self.per_layer_projection(first_prediction)
        first_prediction = self.post_per_layer_input_norm(first_prediction)
        corrected_predictions[1:] += first_prediction

        outputs = (corrected_predictions,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class MiniGPT4OPreTrainedModel(PreTrainedModel):
    config: MiniGPT4OConfig
    base_model_prefix = ""
    _no_split_modules = ["MiniGPT4OTextDecoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MiniGPT4OTextScaledWordEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, embed_scale: float = 1.0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_scale = embed_scale
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        embeddings = nn.functional.embedding(input_ids, self.weight, self.padding_idx, None, 2.0, False, False)
        return embeddings * self.embed_scale


@auto_docstring(custom_intro="The base MiniGPT4O language model without a language modeling head.")
class MiniGPT4OTextModel(PreTrainedModel):
    config: MiniGPT4OTextConfig

    def __init__(self, config: MiniGPT4OTextConfig):
        super().__init__(config)

        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_tokens_per_layer = MiniGPT4OTextScaledWordEmbedding(
            config.vocab_size_per_layer_input,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            config.pad_token_id,
            embed_scale=config.hidden_size_per_layer_input**0.5,
        )

        self.per_layer_model_projection = nn.Linear(
            self.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )

        self.per_layer_projection_norm = MiniGPT4ORMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [MiniGPT4OTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = MiniGPT4ORMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.altup_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.hidden_size, bias=False) for _ in range(1, self.config.altup_num_inputs)]
        )

        self.register_buffer("per_layer_projection_scale", torch.tensor(self.hidden_size**-0.5), persistent=False)
        self.register_buffer("per_layer_input_scale", torch.rsqrt(torch.tensor(2.0)), persistent=False)
        self.rotary_emb = MiniGPT4ORotaryEmbedding(config=config)

        # Local RoPE for sliding attention
        config_local = copy.deepcopy(config)
        config_local.rope_theta = config.rope_local_base_freq
        config_local.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = MiniGPT4ORotaryEmbedding(config=config_local)

        # Initialize weights
        self.post_init()

    def get_per_layer_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_layer_projection: torch.Tensor = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection *= self.per_layer_projection_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (torch.Tensor, *optional*, defaults to None):
            Pre-computed per-layer embeddings. If None, they are derived from input_ids if provided.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
            per_layer_inputs = self.get_per_layer_inputs(input_ids)

        per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        # embed positions
        hidden_states_0 = inputs_embeds

        # Initialize RoPE embeddings
        position_embeddings_global = self.rotary_emb(hidden_states_0, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states_0, position_ids)

        # Expand hidden_states to support per-layer inputs
        target_magnitude = torch.mean(hidden_states_0**2, dim=-1, keepdim=True) ** 0.5
        epsilon_tensor = torch.tensor(1e-5)

        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.config.altup_num_inputs):
            altup_proj = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_type = self.config.layer_types[decoder_layer.layer_idx]
            causal_mask = causal_mask_mapping[layer_type]
            per_layer_input = per_layer_inputs[:, :, decoder_layer.layer_idx, :]

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global,
                position_embeddings_local,
                per_layer_input,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Per-layer inputs to single output
        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True) ** 0.5
        temp_hidden_states = [hidden_states[0]]
        for i in range(1, self.config.altup_num_inputs):
            altup_unemb_proj: torch.Tensor = self.altup_unembed_projections[i - 1](hidden_states[i])
            current_hidden_state = altup_unemb_proj.to(dtype=hidden_states_0.dtype, device=target_magnitude.device)
            new_magnitude = torch.mean(current_hidden_state**2, dim=-1, keepdim=True)
            new_magnitude = torch.sqrt(torch.maximum(new_magnitude, epsilon_tensor.to(target_magnitude.device)))
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring(custom_intro="The base MiniGPT4O language model with a language modeling head.")
class MiniGPT4OForCausalLM(PreTrainedModel):
    config: MiniGPT4OTextConfig
    base_model_prefix = "model"

    def __init__(self, config: MiniGPT4OTextConfig):
        super().__init__(config)
        self.model = MiniGPT4OTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> MiniGPT4OCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        return MiniGPT4OCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MiniGPT4OMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Union[MiniGPT4OAudioConfig, MiniGPT4OVisionConfig],
        text_config: MiniGPT4OTextConfig,
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.multimodal_hidden_size)
        self.hard_embedding_norm = MiniGPT4ORMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.soft_embedding_norm = MiniGPT4ORMSNorm(self.multimodal_hidden_size, eps=self.eps)
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_post_projection_norm = MiniGPT4ORMSNorm(self.text_hidden_size, eps=self.eps, with_scale=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            hard_emb = self.embedding(input_ids - self.vocab_offset)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(emb_norm_proj)


@auto_docstring(
    custom_intro="""
    The base MiniGPT4O model comprising a vision backbone, an audio backbone, and a language model without a
    language modeling head.
    """
)
class MiniGPT4OModel(PreTrainedModel):
    config: MiniGPT4OConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiniGPT4OTextDecoderLayer"]

    def __init__(self, config: MiniGPT4OConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input
        
        # Initialize language model
        self.language_model = MiniGPT4OTextModel(config.text_config)
        
        # Initialize vision and audio towers
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        
        # Initialize embedders
        self.embed_vision = MiniGPT4OMultimodalEmbedder(config.vision_config, config.text_config)
        self.embed_audio = MiniGPT4OMultimodalEmbedder(config.audio_config, config.text_config)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        ).last_hidden_state
        # Convert from (batch, channels, height, width) to (batch, height * width, channels) where:
        # height == width and height * width == MiniGPT4OConfig.vision_soft_tokens_per_image.
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)
        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **lm_kwargs,
    ) -> MiniGPT4OModelOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # Prepare per-layer inputs from inputs_ids
            per_layer_inputs_mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
            per_layer_inputs_tokens = torch.where(per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))
            per_layer_inputs = self.language_model.get_per_layer_inputs(per_layer_inputs_tokens)

            # Handle vision tokens (>= embed_vision.vocab_offset and < embed_audio.vocab_offset)
            vision_mask = torch.logical_and(
                input_ids >= self.embed_vision.vocab_offset, input_ids < self.embed_audio.vocab_offset
            )
            dummy_vision_token_id = self.embed_vision.vocab_offset + self.embed_vision.vocab_size - 1
            vision_input_ids = torch.where(vision_mask, input_ids, dummy_vision_token_id).to(inputs_embeds.device)
            vision_embeds = self.embed_vision(input_ids=vision_input_ids)
            expanded_vision_mask = vision_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_vision_mask, vision_embeds, inputs_embeds)

            # Handle audio tokens (>= embed_audio.vocab_offset)
            audio_mask = input_ids >= self.embed_audio.vocab_offset
            dummy_audio_token_id = self.embed_audio.vocab_offset + self.embed_audio.vocab_size - 1
            audio_input_ids = torch.where(audio_mask, input_ids, dummy_audio_token_id).to(inputs_embeds.device)
            audio_embeds = self.embed_audio(input_ids=audio_input_ids)
            expanded_audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = torch.where(expanded_audio_mask, audio_embeds, inputs_embeds)
        else:
            per_layer_inputs = None

        # Merge text and images
        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text and "
                    f"{image_features.shape[0] * image_features.shape[1]} tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Merge text and audio
        audio_features = None
        if input_features is not None and input_features_mask is not None:
            audio_features, audio_mask = self.get_audio_features(input_features, ~input_features_mask)

            # The MiniGPT4OProcessor expects all audio will be 30s in length and inserts 188 audio soft tokens into the
            # text to account for this. However, the audio preprocessing and encoder do not guarantee they will
            # produce 188 soft tokens; they will produce at most that many tokens, but they may produce fewer tokens
            # depending on the length of the longest audio input in the batch. When we encounter this situation, we pad
            # the audio feature out to 188 soft tokens with the embedding of the last token in the embed_audio vocab.
            audio_padding_toks = torch.tensor([[self.config.text_config.vocab_size - 1]], dtype=torch.long, device=audio_features.device)
            audio_padding_embs = self.embed_audio(input_ids=audio_padding_toks)
            audio_features = torch.where(audio_mask.unsqueeze(-1), audio_padding_embs, audio_features)

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
            extra_padding_tokens = self.config.audio_soft_tokens_per_image - audio_seq_len
            extra_padding_features = audio_padding_embs.expand(audio_batch_size, extra_padding_tokens, audio_embed_dim)

            audio_features = torch.cat((audio_features, extra_padding_features), dim=1)

            if input_ids is None:
                special_audio_mask = inputs_embeds == self.embed_audio(
                    input_ids=torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_audio_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
                special_audio_mask = special_audio_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_audio_mask].numel() != audio_features.numel():
                audio_tokens_in_text = (special_audio_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of audio input features does not match number of special audio tokens in the input text. "
                    f"Got {audio_tokens_in_text} audio tokens in the text and "
                    f"{audio_features.shape[0] * audio_features.shape[1]} tokens from audio embeddings."
                )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_features)

        outputs = self.language_model(
            input_ids=None,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        return MiniGPT4OModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )

    def get_audio_features(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the last hidden state from the audio encoder into language model space.

        Args:
            input_features (`torch.FloatTensor]` of shape `(num_images, seq_length, num_features)`):
               The tensors corresponding to the input audio.
            input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
               The attention mask for the input audio.

        Returns:
            audio_features (`torch.Tensor`): Audio feature tensor of shape `(num_images, audio_length, embed_dim)`).
        """
        audio_outputs, audio_mask = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(inputs_embeds=audio_outputs), audio_mask

    def _update_causal_mask(self, **super_kwargs):
        raise AttributeError("We don't want to inherit it")


@auto_docstring(
    custom_intro="""
    The base MiniGPT4O model comprising a vision backbone, an audio backbone, a language model, and a language modeling
    head.
    """
)
class MiniGPT4OForConditionalGeneration(PreTrainedModel):
    config: MiniGPT4OConfig
    base_model_prefix = "model"

    def __init__(self, config: MiniGPT4OConfig):
        super().__init__(config)
        self.config = config
        self.model = MiniGPT4OModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @property
    def audio_tower(self):
        return self.model.audio_tower

    @property
    def multi_modal_projector(self):
        raise AttributeError("Use embed_vision instead of multi_modal_projector.")

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # text inputs
        pixel_values: Optional[torch.FloatTensor] = None,  # vision inputs
        input_features: Optional[torch.FloatTensor] = None,  # audio inputs
        attention_mask: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> MiniGPT4OCausalLMOutputWithPast:
        r"""
        input_features_mask (torch.Tensor, *optional*, defaults to None):
            The attention mask for the input audio.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in
            `[0, ..., config.text_config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **lm_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if (final_logit_softcapping := self.config.text_config.final_logit_softcapping) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        return MiniGPT4OCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Custom `position_ids` and `pixel_values` handling
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
            "token_type_ids": token_type_ids,
        }

        # If we're in cached decoding stage, multimodal inputs should be None because input ids do not contain special
        # tokens anymore. Otherwise multimodal inputs should be passed to model.
        # NOTE: use_cache=False always needs pixel_values, input_features, and input_features_mask
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask

        return model_inputs


__all__ = [
    "MiniGPT4OAudioConfig",
    "MiniGPT4OConfig",
    "MiniGPT4OForCausalLM",
    "MiniGPT4OForConditionalGeneration",
    "MiniGPT4OModel",
    "MiniGPT4OPreTrainedModel",
    "MiniGPT4OTextConfig",
    "MiniGPT4OTextModel",
    "MiniGPT4OVisionConfig",
]
