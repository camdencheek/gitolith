use std::sync::Arc;

use crate::shard::docs::{DocEnds, DocID};
use crate::shard::suffix::{SuffixBlock, SuffixBlockID};
use crate::shard::ShardID;

use stretto::{
    Cache as StrettoCache, Coster, DefaultCacheCallback, DefaultUpdateValidator, TransparentKey,
    TransparentKeyBuilder,
};

pub type Cache = StrettoCache<
    CacheKey,
    CacheValue,
    TransparentKeyBuilder<CacheKey>,
    CacheValueCoster,
    DefaultUpdateValidator<CacheValue>,
    DefaultCacheCallback<CacheValue>,
    fnv::FnvBuildHasher,
>;

#[derive(Hash, PartialEq, Eq, Debug)]
pub enum CacheKey {
    DocEnds(ShardID),
    DocContent(ShardID, DocID),
    SuffixBlock(ShardID, SuffixBlockID),
}

// Implemented to satisfy TransparentKeyBuilder. Do not actually use this
// because the value returned is meaningless.
impl Default for CacheKey {
    fn default() -> Self {
        Self::DocEnds(ShardID(0))
    }
}

impl TransparentKey for CacheKey {
    fn to_u64(&self) -> u64 {
        use CacheKey::*;
        // First four bits are reserved for the key type.
        // Remaining are available to use for uniqueness
        // within the key. Conflicts are okay, but should
        // be avoided if possible.
        match self {
            DocEnds(shard_id) => (0 << 60) + u64::from(*shard_id),
            DocContent(shard_id, doc_id) => {
                (1 << 60) + (u64::from(*shard_id) << 32) + u64::from(*doc_id)
            }

            SuffixBlock(shard_id, block_id) => {
                (2 << 60) + (u64::from(*shard_id) << 32) + u64::from(*block_id)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum CacheValue {
    DocEnds(Arc<DocEnds>),
    DocContent(Arc<[u8]>),
    SuffixBlock(Arc<SuffixBlock>),
}

impl CacheValue {
    fn size(&self) -> i64 {
        match self {
            CacheValue::DocEnds(e) => e.doc_count() as i64,
            CacheValue::DocContent(c) => c.len() as i64,
            CacheValue::SuffixBlock(_) => SuffixBlock::SIZE_BYTES as i64,
        }
    }
}

pub struct CacheValueCoster();

impl Coster for CacheValueCoster {
    type Value = CacheValue;

    fn cost(&self, val: &Self::Value) -> i64 {
        val.size()
    }
}

pub fn new_cache(max_capacity: u64) -> Cache {
    // TODO tune num_counters
    StrettoCache::builder(10_000, max_capacity as i64)
        .set_coster(CacheValueCoster())
        .set_key_builder(TransparentKeyBuilder::default())
        .set_hasher(fnv::FnvBuildHasher::default())
        .finalize()
        .unwrap()
}
