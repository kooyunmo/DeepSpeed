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

import torch
import torch.distributed as dist
import enum

from typing import Optional, List, Any
from threading import Condition
from concurrent.futures import ThreadPoolExecutor

from deepspeed.utils import event_manager

from .utils import ensure_divisibility


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_USE_WEIGHT_SHARDING = False

def weight_sharding_(flag: bool = True, device = None):
    global _USE_WEIGHT_SHARDING
    _USE_WEIGHT_SHARDING = flag
    if flag:
        _global_weight_shard_context = WeightShardingContext(device=device)

def weight_sharding():
    return _USE_WEIGHT_SHARDING

_global_weight_shard_context = None 
def get_weight_sharding_context() -> "WeightShardingContext":
    return _global_weight_shard_context


class WeightShardingContext:
    def __init__(self, device=None):
        self.reduction_stream = torch.cuda.Stream(device=device)
        self.prefetch_stream = torch.cuda.Stream(device=device)
        self._prefetch_thread_executor = ThreadPoolExecutor(max_workers=1)  # More workers = more memory peak, thus we restrict concurrency degree upto 1
        self._reduce_scatter_thread_executor = ThreadPoolExecutor(max_workers=1)

    def get_group(self):
        # TODO: placeholder for Weight sharding PoC
        return get_model_parallel_group()

    def submit_allgather_params(self, sharding_wrapper: "WeightShardingWrapper"):
        @self._prefetch_thread_executor.submit
        def prefetch_fn():
            event_manager.init_current_thread("ParamPrefetchThread")
            with torch.cuda.stream(self.prefetch_stream):
                sharding_wrapper._allgather_params()

    def submit_reduce_scatter_grads(self, sharding_wrapper: "WeightShardingWrapper"):
        @self._reduce_scatter_thread_executor.submit
        def reduce_scatter_fn():
            event_manager.init_current_thread("ReduceScatterThread")
            with torch.cuda.stream(self.reduction_stream):
                sharding_wrapper._reduce_scatter_grads()


class ParamState(enum.Enum):
    Partitioned = "Partitioned"
    Gathering = "Gathering"
    Gathered = "Gathered"


class GradState(enum.Enum):
    Gathered = "Gathered"
    ReduceScattering = "ReduceScattering"
    ReduceScattered = "ReduceScattered"


class WeightShardingWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module,
                 debug_name: str = "NotSpecified",
                 prev: Optional["WeightShardingWrapper"] = None,
                 next: Optional["WeightShardingWrapper"] = None):
        super().__init__()
        self.module = module
        self.debug_name = debug_name
        self.prev = prev  # public, modifiable
        self.next = next  # public, modifiable

        self.param_state: ParamState = ParamState.Gathered
        self.param_state_cond = Condition()

        self.grad_state: GradState = GradState.Gathered
        self.grad_state_cond = Condition()

        if weight_sharding():
            @module.register_backward_hook
            def after_backward():
                self._presync_grads()
                self._partition_params()

            self._partition_params()
    
    @classmethod
    def wait_for_reduce_scatter(self, wrappers):
        if weight_sharding():
            for wrapper in wrappers:
                wrapper._presync_grads()
            for wrapper in wrappers:
                wrapper._ensure_grad_sync()

    def forward(self, *args, **kwargs):
        if not weight_sharding():
            return self.module(*args, **kwargs)

        self._ensure_params_gathered()
        if self.next: self.next._prefetch_params()
        try:
            return BackwardPreHook.apply(self, self.module(*args, **kwargs))
        finally:
            model._partition_params()

    def _prefetch_params(self):
        should_submit = False
        with self.param_state_cond:
            if self.param_state == ParamState.Partitioned:
                self.param_state = ParamState.Gathering
                self.param_state_cond.notify_all()
                should_submit = True

        if should_submit:
            get_weight_sharding_context().submit_allgather_params(self)

    def _presync_grads(self):
        should_submit = False
        with self.grad_state_cond:
            if self.grad_state == GradState.Gathered:
                self.grad_state = GradState.ReduceScattering
                self.grad_state_cond.notify_all()
                should_submit = True

        if should_submit:
            get_weight_sharding_context().submit_reduce_scatter_grads(self)

    def _ensure_params_gathered(self):
        with event_manager.timespan("ensure_params_gathered", data={"debug_name": self.debug_name}):
            self._prefetch_params()
            with self.param_state_cond:
                while self.param_state != ParamState.Gathered:
                    self.param_state_cond.wait()
    
    def _ensure_grad_sync(self):
        with event_manager.timespan("ensure_grad_sync", data={"debug_name": self.debug_name}):
            self._presync_grads()
            with self.grad_state_cond:
                while self.grad_state != GradState.ReduceScattered:
                    self.grad_state_cond.wait()

    def _partition_params(self):
        assert self.param_state == ParamState.Gathered
        with self.param_state_cond:
            with torch.no_grad():
                group = get_weight_sharding_context().get_group()
                num_shards, rank = dist.get_world_size(group=group), dist.get_rank(group=group)
                for param in self.module.parameters():
                    assert param.size(0) % num_shards == 0
                    sliced_size = param.size(0) // num_shards
                    # Need to clone because slice returns just a sliced "view" of tensor.
                    param.data = param.data[rank*sliced_size:(rank+1)*sliced_size].clone()
            self.param_state = ParamState.Partitioned
            self.param_state_cond.notify_all()

    def _allgather_params(self):
        assert self.param_state != ParamState.Gathered
        try:
            with event_manager.timespan("allgather_params", data={"debug_name": self.debug_name}), torch.no_grad():
                group = get_weight_sharding_context().get_group()
                num_shards, rank = dist.get_world_size(group=group), dist.get_rank(group=group)

                works = []
                for param in self.module.parameters():
                    new_param_data = torch.empty(size=(param.size(0)*num_shards, ) + param.size()[1:], dtype=param.dtype, device=param.device)
                    work = dist.all_gather(list(new_param_data.split(sliced_size, dim=0)), param.data, group=group, async_op=True)
                    works.append(work)
                    param.data = new_param_data

                for work in works:
                    work.wait()
                torch.cuda.current_stream().synchronize()
        finally:
            with self.param_state_cond:
                self.param_state = ParamState.Gathered
                self.param_state_cond.notify_all()

    def _reduce_scatter_grads(self):
        assert self.grad_state != GradState.ReduceScattered
        try:
            with event_manager.timespan("reduce_scatter_grads", data={"debug_name": self.debug_name}), torch.no_grad():
                group = get_weight_sharding_context().get_group()
                num_shards, rank = dist.get_world_size(group=group), dist.get_rank(group=group)
                works = []
                for param, new_param_data in zip(self.module.parameters(), new_param_list):
                    assert param.grad is not None
                    assert not param.grad.is_sparse()
                    slice_size = param.size(0) // num_shards
                    new_grad = torch.empty(size=(slice_size, ) + param.size()[1:], dtype=param.dtype, device=param.device)
                    work = dist.reduce_scatter(new_grad, list(param.grad.split(slice_size, dim=0)), group=group, async_op=True)
                    works.append(work)
                    param.grad = new_grad

                for work in works:
                    work.wait()
                torch.cuda.current_stream().synchronize()
        finally:
            with self.grad_state_cond:
                self.grad_state = ParamState.ReduceScattered
                self.grad_state_cond.notify_all()

    
class BackwardPreHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, wrapper: WeightShardingWrapper, ret_val) -> Any:
        cx.wrapper = wrapper
        return ret_val

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        wrapper = ctx.wrapper
        wrapper._ensure_params_gathered()
        if wrapper.prev: wrapper.prev._prefetch_params()
        return grad_output
 

def initialize_model_parallel(model_parallel_size_):
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel grous as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel with size {}'.format(
            model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = min(model_parallel_size_, world_size)
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank % model_parallel_size):
            _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(world_size // model_parallel_size):
        ranks = range(i * model_parallel_size,
                      (i + 1) * model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if i == (rank // model_parallel_size):
            _MODEL_PARALLEL_GROUP = group

def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zeor
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None

