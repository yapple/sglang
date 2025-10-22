import logging
import os

import torch
import torch.distributed
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
)
from aibrix_kvcache.common.absl_logging import getLogger, log_every_n_seconds, log_if
from aibrix_kvcache_storage import AibrixKVCacheStorage
from torch.distributed import Backend, ProcessGroup

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def setup():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "63886"


class AIBrixKVCacheStorageTest:
    def __init__(self):
        self.config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            is_mla_model=False,
            is_page_first_layout=True,
            model_name="test",
        )
        self.head_num = 1
        self.layer_num = 64
        self.head_dim = 128

    def init_aibrix_kv_cache_storage(self, page_size: int):
        logger.info(f"page_size: {page_size}")
        self.page_size = page_size
        self.mem_pool = MHATokenToKVPool(
            1024,
            page_size,
            torch.float16,
            self.head_num,
            self.head_dim,
            self.layer_num,
            "cpu",
            False,
            0,
            self.layer_num,
        )
        self.mem_pool_host = MHATokenToKVPoolHost(
            self.mem_pool, 2, 0, page_size, "page_first"
        )
        self.aibrix_kvcache = AibrixKVCacheStorage(self.config, self.mem_pool_host)
        self.target_shape = (
            2,
            self.layer_num,
            self.page_size,
            self.head_num,
            self.head_dim,
        )

    def test_with_page_size(self):
        for page_size in range(1, 3):
            self.init_aibrix_kv_cache_storage(page_size)
            batch_size = 2
            query_length = batch_size * 2
            partial = batch_size
            rand_tensor = [
                torch.rand(self.target_shape, dtype=torch.float16)
                for _ in range(query_length)
            ]
            keys = ["hash" + str(i) for i in range(query_length)]
            partial_keys = keys[batch_size:query_length]
            assert self.aibrix_kvcache.batch_exists(keys) == 0
            assert self.aibrix_kvcache.batch_set(keys, rand_tensor)
            get_tensor = [
                torch.rand(self.target_shape, dtype=torch.float16).flatten()
                for _ in range(query_length)
            ]
            self.aibrix_kvcache.batch_get(keys, get_tensor)
            for i in range(query_length):
                assert torch.equal(get_tensor[i], rand_tensor[i].flatten())
            ret = self.aibrix_kvcache.batch_exists(keys)
            assert self.aibrix_kvcache.batch_exists(keys) == query_length
            assert self.aibrix_kvcache.batch_exists(partial_keys) == partial
            partial_get_tensor = [
                torch.rand(self.target_shape, dtype=torch.float16).flatten()
                for _ in range(partial)
            ]
            self.aibrix_kvcache.batch_get(partial_keys, partial_get_tensor)
            for i in range(partial):
                assert torch.equal(
                    partial_get_tensor[i], rand_tensor[i + partial].flatten()
                )
            log_every_n_seconds(
                logger,
                logging.INFO,
                self.aibrix_kvcache.kv_cache_manager.metrics.summary(),
                1,
            )

    def test_external_memory_region(self):
        self.init_aibrix_kv_cache_storage(page_size=16)
        tokens = self.page_size * 2
        keys = [f"key{i}" for i in range(0, tokens // self.page_size)]
        host_indices = self.mem_pool_host.alloc(tokens)
        assert host_indices != None, "alloc host indices failed"
        logger.info(host_indices)
        self.aibrix_kvcache.batch_set_v1(keys, host_indices)


if __name__ == "__main__":
    setup()
    test = AIBrixKVCacheStorageTest()
    test.test_with_page_size()
    test.test_external_memory_region()
