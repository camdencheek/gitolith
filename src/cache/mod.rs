use super::shard::docs::{CompressedDocEnds, DocEnds, DocID};
use super::shard::ShardID;
use stretto::{
    Cache as StrettoCache, CacheBuilder, CacheError, TransparentKey, TransparentKeyBuilder,
};

pub type Cache = StrettoCache<CacheKey, CacheValue, TransparentKeyBuilder<CacheKey>>;

#[derive(Hash, PartialEq, Eq, Default)]
pub enum CacheKey {
    #[default]
    Unknown,
    DocEnds(ShardID),
    CompressedDocEnds(ShardID),
    DocContent(ShardID, DocID),
}

impl TransparentKey for CacheKey {
    fn to_u64(&self) -> u64 {
        match self {
            // First byte is for the key type, rest is for identification within the key type
            CacheKey::Unknown => 0,
            CacheKey::DocEnds(shard_id) => (1 << 56) + u16::from(*shard_id) as u64,
            CacheKey::CompressedDocEnds(shard_id) => (2 << 56) + u16::from(*shard_id) as u64,
            CacheKey::DocContent(shard_id, doc_id) => {
                (3 << 56) + ((u16::from(*shard_id) as u64) << 32) + (u32::from(*doc_id) as u64)
            }
        }
    }
}

pub enum CacheValue {
    DocEnds(DocEnds),
    CompressedDocEnds(CompressedDocEnds),
    DocContent(Vec<u8>),
}

pub fn new_cache(max_size: i64) -> Result<Cache, CacheError> {
    let average_entry_size = 32 * 1024;

    // As recommended in the documentation
    let num_counters = max_size as usize / average_entry_size * 10_000;

    CacheBuilder::new_with_key_builder(
        num_counters,
        max_size,
        TransparentKeyBuilder::<CacheKey>::default(),
    )
    .finalize()
}
