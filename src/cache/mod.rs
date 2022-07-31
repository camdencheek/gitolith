use moka::sync::{Cache as MokaCache, CacheBuilder};

use crate::shard::suffix::{CompressedTrigramPointers, SuffixBlock, SuffixBlockID};

use super::shard::docs::{CompressedDocEnds, DocEnds, DocID};
use super::shard::ShardID;
use std::sync::Arc;

pub type Cache = MokaCache<CacheKey, CacheValue>;

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
    fn size(&self) -> u32 {
        match self {
            CacheValue::DocEnds(e) => e.doc_count() as u32,
            CacheValue::DocContent(c) => c.len() as u32,
            CacheValue::SuffixBlock(_) => SuffixBlock::SIZE_BYTES as u32,
            CacheValue::TrigramPointers(p) => p.size_in_bytes() as u32,
        }
    }
}

pub fn new_cache(max_capacity: u64) -> Cache {
    CacheBuilder::new(max_capacity).build()
}
