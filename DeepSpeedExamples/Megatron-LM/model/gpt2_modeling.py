# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT-2 model."""

import torch
import torch.nn.functional as F

from utils import print_rank_0
import mpu


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


def get_deepspeed_config(args):
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
        from deepspeed import DeepSpeedConfig
        return DeepSpeedConfig(args.deepspeed_config)
    else:
        raise RuntimeError('deepspeed_config is not found in args.')


def get_sparse_attention_config(args, num_heads):
    if args.deepspeed_sparse_attention:
        ds_config = get_deepspeed_config(args)
        if hasattr(ds_config,
                   'sparse_attention') and ds_config.sparse_attention:
            sa_config = ds_config.sparse_attention
            sa_mode = sa_config.get('mode')
            if (sa_mode == 'dense'):
                from deepspeed.ops.sparse_attention import DenseSparsityConfig as STConfig
            elif (sa_mode == 'fixed'):
                from deepspeed.ops.sparse_attention import FixedSparsityConfig as STConfig
            elif (sa_mode == 'bigbird'):
                from deepspeed.ops.sparse_attention import BigBirdSparsityConfig as STConfig
            elif (sa_mode == 'bslongformer'):
                from deepspeed.ops.sparse_attention import BSLongformerSparsityConfig as STConfig
            elif (sa_mode == 'variable'):
                from deepspeed.ops.sparse_attention import VariableSparsityConfig as STConfig
            else:
                raise NotImplementedError(
                    f'Given sparsity mode, {sa_mode}, has not been implemented yet!'
                )
            del sa_config['mode']
            return STConfig(num_heads=num_heads, **sa_config)
        else:
            from deepspeed.ops.sparse_attention import FixedSparsityConfig as STConfig
            print(
                'deepspeed sparse attention is not set; Fixed sparsity is used as default.'
            )
            return STConfig(num_heads=num_heads)
    else:
        return None


class GPT2Model(torch.nn.Module):
    """GPT-2 Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 args,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 ):

        super(GPT2Model, self).__init__()

        self.parallel_output = parallel_output

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                      hidden_size)
        # Initialize the position embeddings.
        init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        self.sparse_attention_config = get_sparse_attention_config(args, num_attention_heads) #TODO

        # Transformer
        self.transformer = mpu.GPT2ParallelTransformer(self.sparse_attention_config,
                                                       num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers
                                                       ) #TODO

    def forward(self, input_ids, position_ids, attention_mask):

        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        # Transformer.
        transformer_output = self.transformer(embeddings, attention_mask)

        # Parallel logits.
        transformer_output_parallel = mpu.copy_to_model_parallel_region(
            transformer_output)
        logits_parallel = F.linear(transformer_output_parallel,
                                   self.word_embeddings.weight)

        if self.parallel_output:
            return logits_parallel

        return mpu.gather_from_model_parallel_region(logits_parallel)


def gpt2_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params
