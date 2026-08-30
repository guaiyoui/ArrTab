# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers.cache_utils import EncoderDecoderCache, DynamicCache

class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, num_beams=1, num_return_sequences=1, opt_info={}):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        # return super().generate(
        #     input_ids=input_ids.view(input_ids.size(0), -1),
        #     attention_mask=attention_mask.view(attention_mask.size(0), -1),
        #     max_length=max_length,
        #     num_beams=num_beams,
        #     num_return_sequences=num_return_sequences,
        #     opt_info = opt_info
        # )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        answer_states = []
        query_passage_states = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)

            score_answer_state = mod.layer[1].EncDecAttention.score_input_1
            bsz, num_answers, emb_size = score_answer_state.size()
            score_answer_state = score_answer_state.view(bsz, 1, num_answers, emb_size)
            answer_states.append(score_answer_state)

            score_query_passage_state = mod.layer[1].EncDecAttention.score_input_2
            bsz, num_passages, emb_size = score_query_passage_state.size()
            score_query_passage_state = score_query_passage_state.view(bsz, 1, num_passages, emb_size)
            query_passage_states.append(score_query_passage_state)

        answer_states = torch.cat(answer_states, dim=1)
        query_passage_states = torch.cat(query_passage_states, dim=1)

        score_input_states = {
            'answer_states':answer_states,
            'query_passage_states':query_passage_states
        }

        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores , score_input_states

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        self.embed_tokens = encoder.embed_tokens
        self.main_input_name = "input_ids"
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                *args
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(*args, **kwargs)
        return output

    # def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
    #     if self.use_checkpoint and self.training:
    #         kwargs = {k: v for k, v in kwargs.items() if v is not None}
    #         def custom_forward(*inputs):
    #             output = self.module(*inputs, **kwargs)
    #             empty = torch.tensor(
    #                 [],
    #                 dtype=torch.float,
    #                 device=output[0].device,
    #                 requires_grad=True)
    #             output = tuple(x if x is not None else empty for x in output)
    #             return output

    #         output = torch.utils.checkpoint.checkpoint(
    #             custom_forward,
    #             hidden_states,
    #             attention_mask,
    #             position_bias
    #         )
    #         output = tuple(x if x.size() != 0 else None for x in output)
    #     else:
    #         output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
    #     return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


import torch
import torch.nn.functional as F
try:
    import transformers
    from transformers.cache_utils import EncoderDecoderCache, DynamicCache
except ImportError:
    # 为了向后兼容，如果没有这些类就定义空的占位符
    EncoderDecoderCache = None
    DynamicCache = None


def cross_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    # past_key_value=None,
    past_key_values=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
    cache_position=None,
):
    """
    Compatible cross-attention forward for Transformers 4.52+ that handles cache variations.
    """
    assert key_value_states is not None, "Cross-attention requires key_value_states"

    batch_size, seq_len, _ = hidden_states.size()
    n_heads = self.n_heads

    # 兼容性处理：获取每个头的维度
    if hasattr(self, 'd_kv'):
        head_dim = self.d_kv
    elif hasattr(self, 'head_dim'):
        head_dim = self.head_dim
    elif hasattr(self, 'inner_dim'):
        head_dim = self.inner_dim // n_heads
    else:
        # 最后的备选方案
        head_dim = self.config.d_kv if hasattr(self, 'config') else 64

    kv_seq_len = key_value_states.size(1)

    # Query projection
    query_states = self.q(hidden_states)
    query_states = query_states.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

    # Key/Value handling with cache compatibility
    key_states = None
    value_states = None

    # 处理不同类型的缓存
    # if past_key_values is not None:
    #     # Transformers 4.52+ 使用新的缓存系统
    #     if hasattr(past_key_values, 'get_cross_attention_cache'):
    #         # 新的 EncoderDecoderCache API
    #         cached_kv = past_key_values.get_cross_attention_cache()
    #         if cached_kv is not None and len(cached_kv) >= 2:
    #             key_states, value_states = cached_kv[0], cached_kv[1]
    #     elif hasattr(past_key_values, 'cross_attention_cache'):
    #         # 检查 cross_attention_cache 属性
    #         if past_key_values.cross_attention_cache and len(past_key_values.cross_attention_cache) >= 2:
    #             key_states, value_states = past_key_values.cross_attention_cache[0], past_key_values.cross_attention_cache[1]
    #     elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) >= 2:
    #         # 传统的 tuple/list 格式
    #         # key_states, value_states = past_key_values[0], past_key_values[1]
    #         key_states, value_states = past_key_values[1]
    #     elif EncoderDecoderCache and isinstance(past_key_values, EncoderDecoderCache):
    #         # 特殊处理 EncoderDecoderCache
    #         if hasattr(past_key_values, 'cross_attention_cache') and past_key_values.cross_attention_cache:
    #             key_states, value_states = past_key_values.cross_attention_cache

    # 如果没有缓存的 key/value，重新计算
    if key_states is None or value_states is None:
        key_states = self.k(key_value_states)
        value_states = self.v(key_value_states)

        key_states = key_states.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, n_heads, head_dim).transpose(1, 2)

    # 计算注意力分数
    print("===================================")
    print(type(query_states), type(key_states))
    print(query_states.shape)
    print(query_states, key_states)
    attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1))

    # 缩放分数
    attention_scores = attention_scores / (head_dim ** 0.5)

    # 应用掩码
    if mask is not None:
        # 确保 mask 的形状正确
        if mask.dim() == 2:
            # 如果是 2D mask，扩展到 4D
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            # 如果是 3D mask，扩展到 4D
            mask = mask.unsqueeze(1)
        attention_scores = attention_scores + mask

    # 处理位置偏置
    if position_bias is None and hasattr(self, 'compute_bias'):
        try:
            position_bias = self.compute_bias(seq_len, kv_seq_len)
        except:
            # 如果 compute_bias 失败，尝试其他方法
            if hasattr(self, 'relative_attention_bias') and self.has_relative_attention_bias:
                position_bias = self.relative_attention_bias(seq_len, kv_seq_len)

    if position_bias is not None:
        attention_scores = attention_scores + position_bias

    # 保存注意力分数用于调试或分析（如果需要）
    if getattr(self, "score_storage", None) is None:
        self.score_storage = attention_scores.detach()
        self.score_input_1 = hidden_states.detach()
        self.score_input_2 = key_value_states.detach()

    # Softmax 和 dropout
    attention_probs = F.softmax(attention_scores.float(), dim=-1).type_as(attention_scores)

    # 应用 dropout
    dropout_p = getattr(self, 'dropout', 0.0)
    if dropout_p > 0.0:
        attention_probs = F.dropout(attention_probs, p=dropout_p, training=self.training)

    # 应用层级头部掩码（如果提供）
    if layer_head_mask is not None:
        attention_probs = attention_probs * layer_head_mask

    # 计算输出
    context_layer = torch.matmul(attention_probs, value_states)
    context_layer = context_layer.transpose(1, 2).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (n_heads * head_dim,)
    context_layer = context_layer.view(new_context_layer_shape)

    # 输出投影
    attention_output = self.o(context_layer)

    # 准备输出 - 严格按照 T5 CrossAttention 的输出格式
    # T5 CrossAttention 的标准输出格式是：
    # (attention_output, present_key_value_state, attention_scores, position_bias)
    # 其中某些元素可能为 None，但位置必须对应

    outputs = [attention_output]  # 第一个元素：attention output

    # 第二个元素：present_key_value (缓存)
    if use_cache:
        if past_key_values is not None:
            if hasattr(past_key_values, 'update_cross_attention_cache'):
                past_key_values.update_cross_attention_cache(key_states, value_states)
                present_key_value = past_key_values
            elif hasattr(past_key_values, 'cross_attention_cache'):
                past_key_values.cross_attention_cache = (key_states, value_states)
                present_key_value = past_key_values
            else:
                present_key_value = (key_states, value_states)
        else:
            if EncoderDecoderCache:
                try:
                    new_cache = EncoderDecoderCache()
                    new_cache.cross_attention_cache = (key_states, value_states)
                    present_key_value = new_cache
                except:
                    present_key_value = (key_states, value_states)
            else:
                present_key_value = (key_states, value_states)
    else:
        present_key_value = None

    outputs.append(present_key_value)

    # 第三个元素：attention scores (只有在 output_attentions=True 时才添加)
    if output_attentions:
        outputs.append(attention_probs)

    # 第四个元素：position_bias (对于交叉注意力，通常存在)
    # 注意：T5 的交叉注意力总是期望 position_bias 作为输出的一部分
    if position_bias is not None:
        outputs.append(position_bias)
    else:
        # 如果没有 position_bias 但模型期望，添加 None
        # 这确保了输出tuple的长度正确
        if hasattr(self, 'has_relative_attention_bias') and self.has_relative_attention_bias:
            outputs.append(None)
        elif len(outputs) < 4 and (output_attentions or use_cache):
            # 确保输出长度符合T5的预期
            outputs.append(None)

    return tuple(outputs)

def cross_attention_forward_ori(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores
        self.score_input_1 = input
        self.score_input_2 = kv

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

class RetrieverConfig(transformers.BertConfig):

    def __init__(self,
                 indexing_dimension=768,
                 apply_question_mask=False,
                 apply_passage_mask=False,
                 extract_cls=False,
                 passage_maxlength=200,
                 question_maxlength=40,
                 projection=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls=extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection

class Retriever(transformers.PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, \
            'If no projection then indexing dimension must be equal to 768'
        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            self.proj = nn.Linear(
                self.model.config.hidden_size,
                self.config.indexing_dimension
            )
            self.norm = nn.LayerNorm(self.config.indexing_dimension)
        self.loss_fct = torch.nn.KLDivLoss()


    def calc_score(self, question_output, passage_output):
        score = torch.matmul(question_output, passage_output.t())
        return score

    def forward(self,
                question_ids,
                question_mask,
                passage_ids,
                passage_mask,
                gold_score=None,
                encode_only=False,
                pos_idxes_per_question=None
        ):
        question_encoded = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        passage_encoded = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        if encode_only:
            return question_encoded, passage_encoded

        score = self.calc_score(question_encoded, passage_encoded)
        _, max_idxs = torch.max(score, 1)
        correct_predictions_count = ((max_idxs == pos_idxes_per_question).sum())
        score = score / np.sqrt(question_encoded.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return score, loss, correct_predictions_count

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids,
            attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)
