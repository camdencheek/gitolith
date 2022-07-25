use super::shard::docs::{CompressedDocEnds, DocEnds, DocID};
use super::shard::ShardID;
use stretto::{Cache, CacheBuilder, TransparentKey, TransparentKeyBuilder};

enum CacheKey {
    DocEnds(ShardID),
    CompressedDocEnds(ShardID),
    DocContent(ShardID, DocID),
}

impl TransparentKey for CacheKey {
    fn to_u64(&self) -> u64 {
        match self {
            DocEnds(shard_id) => 0 << 32 + shard_id,
        }
    }
}

enum CacheValue {
    DocEnds(DocEnds),
    CompressedDocEnds(CompressedDocEnds),
    DocContent(Vec<u8>),
}

pub fn new_cache(max_size: i64) -> Cache<CacheKey, CacheValue> {
    let average_entry_size = 32 * 1024;

    // As recommended in the documentation
    let num_counters = max_size / average_entry_size * 10_000;

    CacheBuilder::new_with_key_builder(num_counters, max_size, TransparentKeyBuilder::default());
}
