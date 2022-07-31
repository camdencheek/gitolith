use std::{io, sync::Arc};

use crate::cache::{Cache, CacheKey, CacheValue};

use super::{
    content::ContentStore,
    docs::{DocEnds, DocID, DocStore},
    suffix::{
        CompressedTrigramPointers, ReadTrigramPointersError, SuffixArrayStore, SuffixBlock,
        SuffixBlockID, SuffixIdx,
    },
    Shard, ShardID,
};

#[derive(Clone)]
pub struct CachedShard {
    id: ShardID,
    shard: Shard,
    cache: Cache,
}

impl CachedShard {
    fn new(id: ShardID, shard: Shard, cache: Cache) -> Self {
        Self { id, shard, cache }
    }
}

pub struct CachedDocs {
    shard_id: ShardID,
    docs: DocStore,
    content: ContentStore,
    cache: Cache,
}

impl CachedDocs {
    pub fn doc_ids(&self) -> impl Iterator<Item = DocID> {
        self.docs.doc_ids()
    }

    pub fn read_content(
        &self,
        doc_id: DocID,
        doc_ends: &DocEnds,
    ) -> Result<Arc<Vec<u8>>, Arc<io::Error>> {
        let key = CacheKey::DocContent(self.shard_id, doc_id);
        self.cache
            .try_get_with(key, || -> Result<CacheValue, io::Error> {
                self.docs
                    .read_content(doc_id, doc_ends)
                    .map(|v| CacheValue::DocContent(Arc::new(v)))
            })
            .map(|v| match v {
                CacheValue::DocContent(c) => c,
                _ => panic!("expected CacheValue::DocContent, got {:?}", v),
            })
    }

    pub fn read_doc_ends(&self) -> Result<Arc<DocEnds>, Arc<io::Error>> {
        let key = CacheKey::DocEnds(self.shard_id);
        self.cache
            .try_get_with(key, || -> Result<CacheValue, io::Error> {
                self.docs
                    .read_doc_ends()
                    .map(|v| CacheValue::DocEnds(Arc::new(v)))
            })
            .map(|v| match v {
                CacheValue::DocEnds(c) => c,
                _ => panic!("expected CacheValue::DocEnds, got {:?}", v),
            })
    }

    pub fn num_docs(&self) -> u32 {
        self.docs.num_docs()
    }

    pub fn max_doc_id(&self) -> DocID {
        self.docs.max_doc_id()
    }
}

struct CachedSuffixes {
    shard_id: ShardID,
    suffixes: SuffixArrayStore,
    content: ContentStore,
    cache: Cache,
}

impl CachedSuffixes {
    pub fn new(
        shard_id: ShardID,
        suffixes: SuffixArrayStore,
        content: ContentStore,
        cache: Cache,
    ) -> Self {
        Self {
            shard_id,
            suffixes,
            content,
            cache,
        }
    }

    pub fn block_id_for_suffix(suffix: SuffixIdx) -> SuffixBlockID {
        SuffixArrayStore::block_id_for_suffix(suffix)
    }

    pub fn read_block(&self, block_id: SuffixBlockID) -> Result<Arc<SuffixBlock>, Arc<io::Error>> {
        let key = CacheKey::SuffixBlock(self.shard_id, block_id);
        self.cache
            .try_get_with(key, || -> Result<CacheValue, io::Error> {
                self.suffixes
                    .read_block(block_id)
                    .map(|v| CacheValue::SuffixBlock(Arc::new(*v)))
            })
            .map(|v| match v {
                CacheValue::SuffixBlock(c) => c,
                _ => panic!("expected CacheValue::DocEnds, got {:?}", v),
            })
    }

    pub fn read_trigram_pointers(
        &self,
    ) -> Result<Arc<CompressedTrigramPointers>, Arc<ReadTrigramPointersError>> {
        let key = CacheKey::TrigramPointers(self.shard_id);
        self.cache
            .try_get_with(key, || -> Result<CacheValue, ReadTrigramPointersError> {
                self.suffixes
                    .read_trigram_pointers()
                    .map(|v| CacheValue::TrigramPointers(Arc::new(v)))
            })
            .map(|v| match v {
                CacheValue::TrigramPointers(p) => p,
                _ => panic!("expected CacheValue::DocEnds, got {:?}", v),
            })
    }
}
