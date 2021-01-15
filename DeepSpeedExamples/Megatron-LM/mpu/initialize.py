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


"""Model and data parallel groups."""
import functools
from typing import Optional

import torch
import torch.distributed as dist

from .utils import ensure_divisibility

from deepspeed.runtime.pipe.topology import PipelineParallelGrid, PipeModelDataParallelTopology

# 3D parallelism group: [pipe, data, model]
_PIPELINE_PARALLEL_GRID: Optional[PipelineParallelGrid] = None


def initialize_model_parallel(topology: PipeModelDataParallelTopology):
    global _PIPELINE_PARALLEL_GRID
    assert dist.is_initialized()
    assert not _PIPELINE_PARALLEL_GRID, 'already initialized'
    _PIPELINE_PARALLEL_GRID = PipelineParallelGrid(topology=topology)


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    return _PIPELINE_PARALLEL_GRID is not None


def ensure_initialized(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if not model_parallel_is_initialized():
            raise RuntimeError("model parallelism is not initialized")
        return fn(*args, **kwargs)

    return wrapped


@ensure_initialized
def get_pipe_parallel_rank():
    return _PIPELINE_PARALLEL_GRID.get_pipe_parallel_rank()


@ensure_initialized
def get_pipe_parallel_group():
    return _PIPELINE_PARALLEL_GRID.get_pipe_parallel_group()


@ensure_initialized
def get_pipe_parallel_world_size():
    return _PIPELINE_PARALLEL_GRID.get_pipe_parallel_world_size()


@ensure_initialized
def get_data_parallel_rank():
    return _PIPELINE_PARALLEL_GRID.get_data_parallel_rank()


@ensure_initialized
def get_data_parallel_group():
    return _PIPELINE_PARALLEL_GRID.get_data_parallel_group()


@ensure_initialized
def get_data_parallel_world_size():
    return _PIPELINE_PARALLEL_GRID.get_data_parallel_world_size()


# Note (AC): "slice" dimension in Deepspeed pipe module means model dimension in Megatron-LM
#            On the other hand, model dimension in Deepspeed pipe module is actually model dimension times pipeline dimension.
@ensure_initialized
def get_model_parallel_rank():
    return _PIPELINE_PARALLEL_GRID.get_slice_parallel_rank()


@ensure_initialized
def get_model_parallel_group():
    return _PIPELINE_PARALLEL_GRID.get_slice_parallel_group()


@ensure_initialized
def get_model_parallel_world_size():
    return _PIPELINE_PARALLEL_GRID.get_slice_parallel_world_size()


@ensure_initialized
def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zeor
    in the model parallel group."""
    global_rank = dist.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size

@ensure_initialized
def get_data_parallel_src_rank():
    global_rank = dist.get_rank()
    local_world_size = get_model_parallel_world_size() * get_data_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size + get_model_parallel_rank()

def destroy_model_parallel():
    """Set the groups to none."""
    global _PIPELINE_PARALLEL_GRID
    _PIPELINE_PARALLEL_GRID = None
