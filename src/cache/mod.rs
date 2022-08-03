use std::sync::Arc;

use crate::shard::docs::{CompressedDocEnds, DocEnds, DocID};
use crate::shard::suffix::{CompressedTrigramPointers, SuffixBlock, SuffixBlockID};
use crate::shard::ShardID;

use stretto::{Cache as StrettoCache, Coster, DefaultKeyBuilder};

pub type Cache = StrettoCache<CacheKey, CacheValue, DefaultKeyBuilder<CacheKey>, CacheValueCoster>;

#[derive(Hash, PartialEq, Eq, Debug)]
pub enum CacheKey {
    DocEnds(ShardID),
    DocContent(ShardID, DocID),
    SuffixBlock(ShardID, SuffixBlockID),
    TrigramPointers(ShardID),
}

#[derive(Clone, Debug)]
pub enum CacheValue {
    DocEnds(Arc<DocEnds>),
    DocContent(Arc<Vec<u8>>),
    SuffixBlock(Arc<SuffixBlock>),
    TrigramPointers(Arc<CompressedTrigramPointers>),
}

impl CacheValue {
    fn size(&self) -> i64 {
        match self {
            CacheValue::DocEnds(e) => e.doc_count() as i64,
            CacheValue::DocContent(c) => c.len() as i64,
            CacheValue::SuffixBlock(_) => SuffixBlock::SIZE_BYTES as i64,
            CacheValue::TrigramPointers(p) => p.size_in_bytes() as i64,
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
        .finalize()
        .unwrap()
}
