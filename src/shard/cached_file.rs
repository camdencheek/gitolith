use std::sync::Arc;

use anyhow::Error;

use crate::{
    cache::{Cache, CacheKey, CacheValue},
    shard::suffix::SuffixBlock,
};

use super::{
    docs::{DocEnds, DocID},
    file::{ShardBackend, ShardFile, ShardHeader, ShardStore},
    suffix::SuffixBlockID,
    ShardID,
};

pub struct CachedShardFile {
    shard_id: ShardID,
    cache: Cache,
    file: ShardFile,
}

impl CachedShardFile {
    pub fn new(shard_id: ShardID, cache: Cache, file: ShardFile) -> Self {
        Self {
            shard_id,
            cache,
            file,
        }
    }
}

impl ShardBackend for CachedShardFile {
    fn header(&self) -> &ShardHeader {
        &self.file.header
    }

    fn read_doc_ends(&self) -> Result<Arc<DocEnds>, Error> {
        let key = CacheKey::DocEnds(self.shard_id);
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::DocEnds(self.file.read_doc_ends()?);
            self.cache.insert(key, v.clone(), 0);
            v
        };

        match value {
            CacheValue::DocEnds(de) => Ok(de),
            _ => unreachable!(),
        }
    }

    fn read_doc(&self, doc_id: DocID, doc_ends: &DocEnds) -> Result<Arc<[u8]>, Error> {
        let key = CacheKey::DocContent(self.shard_id, doc_id);
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::DocContent(self.file.read_doc(doc_id, doc_ends)?);
            self.cache.insert(key, v.clone(), 0);
            v
        };

        match value {
            CacheValue::DocContent(dc) => Ok(dc),
            _ => unreachable!(),
        }
    }

    fn read_suffix_block(&self, block_id: SuffixBlockID) -> Result<Arc<SuffixBlock>, Error> {
        let key = CacheKey::SuffixBlock(self.shard_id, block_id);
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::SuffixBlock(self.file.read_suffix_block(block_id)?);
            self.cache.insert(key, v.clone(), 0);
            v
        };

        match value {
            CacheValue::SuffixBlock(b) => Ok(b),
            _ => unreachable!(),
        }
    }
}
