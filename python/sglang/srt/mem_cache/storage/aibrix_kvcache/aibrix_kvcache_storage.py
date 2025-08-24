import logging
import os
from typing import Any, List, Optional, Sequence, Union

import torch
from aibrix_kvcache import (
    BaseKVCacheManager,
    GroupAwareKVCacheManager,
    KVCacheBlockLayout,
    KVCacheBlockSpec,
    KVCacheConfig,
    KVCacheMetrics,
    KVCacheTensorSpec,
    ModelSpec,
    TokenListView,
    envs,
)
from aibrix_kvcache.cache_handle import KVCacheHandle
from aibrix_kvcache.common.absl_logging import getLogger, log_every_n_seconds, log_if
from aibrix_kvcache.memory import MemoryRegion
from aibrix_kvcache.utils import perf_timer

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)

logger = logging.getLogger(__name__)


class AibrixKVCacheStorage(HiCacheStorage):
    def __init__(self, kv_cache: KVCache):
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        self.page_size = kv_cache.page_size
        self.kv_cache_dtype = kv_cache.dtype
        self.kv_cache = kv_cache
        self.layer_num = self.kv_cache.layer_num
        # self.kv_head_ids = range(tp_rank * tp_size, (tp_rank+ 1) * tp_size) # for tensor parallel
        self.kv_head_ids = range(self.kv_cache.head_num)  # for tensor parallel
        if isinstance(kv_cache, MLATokenToKVPool):
            self.kv_cache_shape = (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache.kv_lora_rank + self.kv_cache.qk_rope_head_dim,
            )
            logger.error(f"MLA is not support")
        elif isinstance(kv_cache, MHATokenToKVPool):
            self.kv_cache_shape = (
                2,
                self.layer_num,
                self.page_size,
                self.kv_cache.head_num,
                self.kv_cache.head_dim,
            )
            print(self.kv_cache_shape)
            if self.kv_cache.start_layer != self.kv_cache.end_layer:
                self.layer_ids = range(
                    self.kv_cache.start_layer, self.kv_cache.end_layer
                )  # for pipeline parallel
            else:
                self.layer_ids = range(self.layer_num + 1)  # for pipeline parallel

            print(self.layer_ids, self.layer_num)
            self.block_spec = KVCacheBlockSpec(
                block_ntokens=self.page_size,
                block_dtype=self.kv_cache_dtype,
                block_layout=KVCacheBlockLayout(KVCacheBlockLayout.NCLD),
                tensor_spec=KVCacheTensorSpec(
                    heads=self.kv_head_ids,
                    layers=self.layer_ids,
                    head_size=self.kv_cache.head_dim,
                ),
            )
            print(self.block_spec)
            config = KVCacheConfig(
                block_spec=self.block_spec, model_spec=ModelSpec(102400)
            )
            self.kv_cache_manager = BaseKVCacheManager(config)

    def prefix_get(
        self,
        prefix: List[int],
        token_ids: List[int],
        target_location: Optional[Any],
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        ids_view = TokenListView(prefix + token_ids)
        status = self.kv_cache_manager.acquire(
            ids_view[: len(prefix)], ids_view[len(prefix) :]
        )
        if status.is_ok():
            num_fetched_tokens, handle = status.value
            kv_blocks = handle.to_tensors()
            logger.info(f"handle {kv_blocks[0].shape}")
            logger.info(f"target {target_location[0].shape}")
            assert len(kv_blocks) == len(target_location)
            for i in range(len(kv_blocks)):
                target_location[i].reshape(kv_blocks[i].shape).copy_(
                    kv_blocks[i]
                ).reshape(target_location[i].shape)
            return target_location

        return None

    def prefix_set(
        self,
        prefix: List[int],
        token_ids: List[int],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        status = self.kv_cache_manager.allocate_for(prefix, token_ids)
        if not status.is_ok():
            logger.error("prefix_set allocate failed")
            return False
        handle = status.value
        tensors = handle.to_tensors()
        if len(tensors) != len(values):
            logger.error("prefix_set allocate not enough")
            return False
        for i in range(len(tensors)):
            tensors[i].reshape(values[i].shape).copy_(values[i]).reshape(
                tensors[i].shape
            )
        ids_view = TokenListView(prefix + token_ids)
        status = self.kv_cache_manager.put(
            ids_view[: len(prefix)], ids_view[len(prefix) :], handle
        )
        if not status.is_ok():
            logger.info(
                f"AIBrix KVCache Storage prefix set failed, error_code {status.error_code}"
            )
            return False
        completed = status.value
        return completed == len(token_ids)

    def prefix_exists(self, prefix: List[int], token_ids: List[int]) -> int:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        assert token_ids != None
        ids_view = TokenListView(prefix + token_ids)
        status = self.kv_cache_manager.exists(
            ids_view[: len(prefix)], ids_view[len(prefix) :]
        )
        if status.is_ok():
            return status.value
        return 0

    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        return None

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        """
        return False

    def exists(self, key: str) -> bool | dict:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        return False
