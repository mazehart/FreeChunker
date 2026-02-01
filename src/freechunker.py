# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""FreeChunker model: Modified from PyTorch XLM-RoBERTa model."""
from src.utils import generate_shifted_matrix
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    logging
)
from .configuration import FreeChunkerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "FacebookAI/xlm-roberta-base"
_CONFIG_FOR_DOC = "FreeChunkerConfig"


# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->FreeChunker
class FreeChunkerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=self.position_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->FreeChunker
class FreeChunkerSelfAttention(nn.Module):  
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,  # Second input stream, required parameter
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Query comes from hidden_states
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # Key and Value come from hidden_states2
        key_layer = self.transpose_for_scores(self.key(hidden_states2))
        value_layer = self.transpose_for_scores(self.value(hidden_states2))

        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # Modified positional encoding handling
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            
            # hidden_states positions are all the first position (0, 0, 0, ...)
            position_ids_l = torch.zeros(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            # hidden_states2 uses normal incremental position sequence (0, 1, 2, 3, ...)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply head mask
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Calculate context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaSdpaSelfAttention with Roberta->FreeChunker
class FreeChunkerSdpaSelfAttention(FreeChunkerSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,  # Second input stream, required parameter
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # If relative positional encoding, output attentions, or head mask are present, fallback to parent implementation
        if (self.position_embedding_type != "absolute" or 
            output_attentions or 
            head_mask is not None):
            return super().forward(
                hidden_states,
                hidden_states2,
                attention_mask,
                head_mask,
                output_attentions,
            )

        # Use optimized implementation of SDPA
        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states2))
        value_layer = self.transpose_for_scores(self.value(hidden_states2))

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,  # For customized tasks, causal mask is not used
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput with Roberta->FreeChunker
class FreeChunkerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


XLM_ROBERTA_SELF_ATTENTION_CLASSES = {
    "eager": FreeChunkerSelfAttention,
    "sdpa": FreeChunkerSdpaSelfAttention,
}


# Copied from transformers.models.roberta.modeling_roberta.RobertaAttention with Roberta->FreeChunker
class FreeChunkerAttention(nn.Module):  
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = XLM_ROBERTA_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, position_embedding_type=position_embedding_type
        )
        self.output = FreeChunkerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,  # Second input stream, required parameter
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            hidden_states2,  # Pass second input stream
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaIntermediate with Roberta->FreeChunker
class FreeChunkerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput with Roberta->FreeChunker
class FreeChunkerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaLayer with Roberta->FreeChunker
class FreeChunkerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FreeChunkerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = FreeChunkerAttention(config, position_embedding_type="absolute")
        self.intermediate = FreeChunkerIntermediate(config)
        self.output = FreeChunkerOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,  # Second input stream, required parameter
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_outputs = self.attention(
            hidden_states,
            hidden_states2,  # Pass second input stream
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaEncoder with Roberta->FreeChunker
class FreeChunkerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([FreeChunkerLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states2: torch.Tensor,  # Second input stream, required parameter
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    hidden_states2,  # Pass second input stream
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    hidden_states2,  # Pass second input stream
                    attention_mask,
                    layer_head_mask,
                )

            hidden_states = layer_outputs[0]

        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaPooler with Roberta->FreeChunker
class FreeChunkerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel with Roberta->FreeChunker
class FreeChunkerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FreeChunkerConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FreeChunkerEmbeddings", "FreeChunkerSelfAttention", "FreeChunkerSdpaSelfAttention"]
    _supports_sdpa = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


XLM_ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FreeChunkerConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XLM_ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
# Copied from transformers.models.roberta.modeling_roberta.RobertaModel with Roberta->FreeChunker, ROBERTA->XLM_ROBERTA
class FreeChunkerModel(FreeChunkerPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _no_split_modules = ["FreeChunkerEmbeddings", "FreeChunkerLayer"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.config.vocab_size = 2
        self.embeddings = FreeChunkerEmbeddings(self.config)
        self.encoder = FreeChunkerEncoder(config)

        self.pooler = FreeChunkerPooler(config) if add_pooling_layer else None  

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        inputs_embeds=None,
        labels=None,
        loss_weights: bool = False,
        input_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        granularities: Optional[list] = None
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        
        # Get input device
        input_device = inputs_embeds.device
        
        # Dimension adaptation: if input dimension is less than 1024, pad with 0
        original_hidden_size = inputs_embeds.shape[-1]
        target_hidden_size = self.config.hidden_size  # 1024
        
        if original_hidden_size < target_hidden_size:
            # Calculate number of dimensions to pad
            padding_size = target_hidden_size - original_hidden_size
            # Pad with 0 on the last dimension
            padding = torch.zeros(inputs_embeds.shape[:-1] + (padding_size,), 
                                device=input_device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, padding], dim=-1)
        
        # Adjust max_power based on sequence length
        sequence_length = inputs_embeds.shape[1]
        
        shifted_matrix = generate_shifted_matrix(sequence_length, device=input_device, granularities=granularities)
        
        # Generate attention mask
        encoder_attention_mask = shifted_matrix.transpose(1, 2)
        encoder_attention_mask = torch.where(encoder_attention_mask == 1.0, 0.0, float('-inf'))[:, None, :, :]
        
        # Fixed input IDs and position IDs
        input_ids = torch.tensor([[0] * shifted_matrix.shape[2]], device=input_device)
        position_ids = torch.tensor([[0] * shifted_matrix.shape[2]], device=input_device)
        
        # Embedding layer processing
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=None,
        )

        # Set second input stream
        encoder_hidden_states = inputs_embeds

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Encoder processing
        sequence_output = self.encoder(
            embedding_output,
            hidden_states2=encoder_hidden_states,  # Second input stream
            attention_mask=encoder_attention_mask,  # Use generated mask
            head_mask=head_mask,
        )

        if original_hidden_size < target_hidden_size:
            
            sequence_output = sequence_output[..., :original_hidden_size]
            # Also truncate inputs_embeds back to original size to match dimensions of sequence_output
            inputs_embeds = inputs_embeds[..., :original_hidden_size]

        shift_matrix = shifted_matrix.transpose(1, 2).squeeze(0)
        # Loss calculation
        loss = None
        if labels is not None:
            emb = sequence_output.view(-1, sequence_output.shape[-1])
            lab = labels.view(-1, labels.shape[-1])
            target = torch.ones(emb.size(0), device=emb.device)
            
            # If weights are provided, use weighted cosine loss
            if loss_weights:
                # Validate weight dimensions
                loss_weights = shift_matrix.sum(dim=1).to(emb.device)
                
                # Calculate unweighted cosine loss
                cos_loss_fn = torch.nn.CosineEmbeddingLoss(reduction='none')
                individual_losses = cos_loss_fn(emb, lab, target)
                
                # Apply weights and calculate weighted average
                weighted_losses = individual_losses * loss_weights
                loss = weighted_losses.sum() / loss_weights.sum()
            else:
                # Use standard cosine loss
                cos_loss = torch.nn.CosineEmbeddingLoss()
                loss = cos_loss(emb, lab, target)
        
        embedding = torch.cat([inputs_embeds, sequence_output], dim=1)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        # embedding = torch.nn.functional.normalize(sequence_output, p=2, dim=-1)
        
        return {
            "loss": loss, 
            "embedding": embedding.squeeze(0), 
            "shift_matrix": shift_matrix
        }

# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
    return incremental_indices.long() + padding_idx


__all__ = [
    "FreeChunkerModel",
    "FreeChunkerPreTrainedModel",
]
