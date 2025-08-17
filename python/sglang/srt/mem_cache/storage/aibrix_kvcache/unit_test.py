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
from aibrix_kvcache_storage import AibrixKVCacheStorage
from torch.distributed import Backend, ProcessGroup

from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool


def setup():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "63886"


class AIBrixKVCacheStorgaeTest:
    def test_with_page_size(self):
        for page_size in range(1, 3):
            batch_size = 2
            head_num = 1
            # layer_num > 1 : the shape in aibrix has some problem need to resolve
            layer_num = 1
            head_dim = 63
            kv_cache = MHATokenToKVPool(
                1024,
                page_size,
                torch.float16,
                head_num,
                head_dim,
                layer_num,
                "cpu",
                False,
            )
            self.aibrix_kvcache = AibrixKVCacheStorage(kv_cache)
            tokens = [i for i in range(page_size * (batch_size + 1))]
            prefix = tokens[:page_size]
            tokens_id = tokens[page_size:]
            target_shape = (head_num, 2, page_size, layer_num, head_dim)
            rand_tensor = [torch.rand(target_shape, dtype=torch.float16)] * batch_size
            assert self.aibrix_kvcache.prefix_exists(prefix, tokens_id) == False
            self.aibrix_kvcache.prefix_set(prefix, tokens_id, rand_tensor)
            get_tensor = [torch.rand(target_shape, dtype=torch.float16)] * batch_size
            self.aibrix_kvcache.prefix_get(prefix, tokens_id, get_tensor)
            for i in range(batch_size):
                assert torch.equal(get_tensor[i], rand_tensor[i])
            assert self.aibrix_kvcache.prefix_exists(prefix, tokens_id) == True


if __name__ == "__main__":
    setup()
    init_distributed_environment()
    initialize_model_parallel()
    test = AIBrixKVCacheStorgaeTest()
    test.test_with_page_size()
