use std::{
    io,
    ops::{Range, RangeInclusive},
    sync::Arc,
};

use crate::cache::{Cache, CacheKey, CacheValue};

use super::{
    content::{ContentIdx, ContentStore},
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
    pub fn new(id: ShardID, shard: Shard, cache: Cache) -> Self {
        Self { id, shard, cache }
    }

    pub fn docs(&self) -> CachedDocs {
        CachedDocs::new(self.id, self.shard.docs.clone(), self.cache.clone())
    }

    pub fn suffixes(&self) -> CachedSuffixes {
        CachedSuffixes::new(
            self.id,
            self.shard.suffixes.clone(),
            self.shard.docs.clone(),
            self.cache.clone(),
        )
    }
}

#[derive(Clone)]
pub struct CachedDocs {
    shard_id: ShardID,
    docs: DocStore,
    cache: Cache,
}

impl CachedDocs {
    pub fn new(shard_id: ShardID, docs: DocStore, cache: Cache) -> Self {
        Self {
            shard_id,
            docs,
            cache,
        }
    }

    pub fn doc_ids(&self) -> impl Iterator<Item = DocID> {
        self.docs.doc_ids()
    }

    pub fn read_content(&self, doc_id: DocID, doc_ends: &DocEnds) -> Arc<Vec<u8>> {
        let key = CacheKey::DocContent(self.shard_id, doc_id);
        let value = self.cache.get_with(key, || -> CacheValue {
            CacheValue::DocContent(Arc::new(
                self.docs
                    .read_content(doc_id, doc_ends)
                    .expect("failed to read doc content"),
            ))
        });

        match value {
            CacheValue::DocContent(dc) => dc,
            _ => unreachable!(),
        }
    }

    pub fn read_doc_ends(&self) -> Arc<DocEnds> {
        let key = CacheKey::DocEnds(self.shard_id);
        let value = self.cache.get_with(key, || -> CacheValue {
            CacheValue::DocEnds(Arc::new(
                self.docs.read_doc_ends().expect("failed to read doc ends"),
            ))
        });

        match value {
            CacheValue::DocEnds(de) => de,
            _ => unreachable!(),
        }
    }

    pub fn num_docs(&self) -> u32 {
        self.docs.num_docs()
    }

    pub fn max_doc_id(&self) -> DocID {
        self.docs.max_doc_id()
    }
}

pub struct CachedSuffixes {
    shard_id: ShardID,
    suffixes: SuffixArrayStore,
    docs: CachedDocs,
    cache: Cache,
}

impl CachedSuffixes {
    pub fn new(
        shard_id: ShardID,
        suffixes: SuffixArrayStore,
        docs: DocStore,
        cache: Cache,
    ) -> Self {
        Self {
            shard_id,
            suffixes,
            docs: CachedDocs::new(shard_id, docs, cache.clone()),
            cache,
        }
    }

    pub fn block_range(suffix_range: Range<SuffixIdx>) -> Range<(SuffixBlockID, usize)> {
        SuffixArrayStore::block_range(suffix_range)
    }

    pub fn block_id_for_suffix(suffix: SuffixIdx) -> (SuffixBlockID, usize) {
        SuffixArrayStore::block_id_for_suffix(suffix)
    }

    pub fn lookup_prefix_range(&self, prefix_range: RangeInclusive<Vec<u8>>) -> Range<SuffixIdx> {
        let trigrams = self.read_trigram_pointers();
        let start_bounds = trigrams.bounds(prefix_range.start()..=prefix_range.start());
        let end_bounds = trigrams.bounds(prefix_range.end()..=prefix_range.end());
        if start_bounds.start == end_bounds.end {
            // Return early if there are no trigrams that match
            return start_bounds.start..end_bounds.end;
        }

        let (start, end) = prefix_range.into_inner();
        self.lookup_prefix_start(start, start_bounds)..self.lookup_prefix_end(end, end_bounds)
    }

    fn lookup_prefix_start(&self, prefix: Vec<u8>, bounds: Range<SuffixIdx>) -> SuffixIdx {
        let doc_ends = self.docs.read_doc_ends();
        let pred = |suffix_idx| {
            // TODO we can probably improve perf here by holding onto the last block or two
            let (block_id, offset) = Self::block_id_for_suffix(suffix_idx);
            let block = self.read_block(block_id);
            let content_idx = block.0[offset];
            let doc_id = doc_ends.find(content_idx);
            let doc_content = self.docs.read_content(doc_id, &doc_ends);
            let doc_content_range = doc_ends.content_range(doc_id);
            let content_end =
                usize::from(doc_content_range.end).max(usize::from(content_idx) + prefix.len());
            let content = &doc_content[usize::from(content_idx)..content_end];
            let prefix_slice: &[u8] = &prefix;
            // TODO up until here, all the logic is the same as lookup_prefix_end.
            // We can probably both deduplicate and run the lookups at the same time
            // to avoid the cost of re-fetching the blocks and content
            return prefix_slice < content;
        };
        self.find_suffix_idx(pred, Some(bounds))
    }

    fn lookup_prefix_end(&self, prefix: Vec<u8>, bounds: Range<SuffixIdx>) -> SuffixIdx {
        let doc_ends = self.docs.read_doc_ends();
        let pred = |suffix_idx| {
            // TODO we can probably improve perf here by holding onto the last block or two
            let (block_id, offset) = Self::block_id_for_suffix(suffix_idx);
            let block = self.read_block(block_id);
            let content_idx = block.0[offset];
            let doc_id = doc_ends.find(content_idx);
            let doc_content = self.docs.read_content(doc_id, &doc_ends);
            let doc_content_range = doc_ends.content_range(doc_id);
            let content_end =
                usize::from(doc_content_range.end).max(usize::from(content_idx) + prefix.len());
            let content = &doc_content[usize::from(content_idx)..content_end];
            let prefix_slice: &[u8] = &prefix;
            // TODO up until here, all the logic is the same as lookup_prefix_end.
            // We can probably both deduplicate and run the lookups at the same time
            // to avoid the cost of re-fetching the blocks and content
            return prefix_slice <= content;
        };
        self.find_suffix_idx(pred, Some(bounds))
    }

    fn find_suffix_idx<T>(&self, mut pred: T, bounds: Option<Range<SuffixIdx>>) -> SuffixIdx
    where
        T: FnMut(SuffixIdx) -> bool,
    {
        let (mut min, mut max) = match bounds {
            Some(r) => (r.start, r.end),
            None => (SuffixIdx(0), SuffixIdx(self.suffixes.sa_len)),
        };

        while min < max {
            let mid = SuffixIdx((u32::from(max) - u32::from(min)) / 2 + u32::from(min));
            if pred(mid) {
                min = mid + SuffixIdx(1)
            } else {
                max = mid
            }
        }
        return min;
    }

    pub fn read_block(&self, block_id: SuffixBlockID) -> Arc<SuffixBlock> {
        let key = CacheKey::SuffixBlock(self.shard_id, block_id);
        let value = self.cache.get_with(key, || -> CacheValue {
            CacheValue::SuffixBlock(Arc::new(
                *self
                    .suffixes
                    .read_block(block_id)
                    .expect("failed to read suffix block"),
            ))
        });

        match value {
            CacheValue::SuffixBlock(b) => b,
            _ => unreachable!(),
        }
    }

    pub fn read_trigram_pointers(&self) -> Arc<CompressedTrigramPointers> {
        let key = CacheKey::TrigramPointers(self.shard_id);
        let value = self.cache.get_with(key, || -> CacheValue {
            CacheValue::TrigramPointers(Arc::new(
                self.suffixes
                    .read_trigram_pointers()
                    .expect("failed to read trigram pointers"),
            ))
        });

        match value {
            CacheValue::TrigramPointers(tp) => tp,
            _ => unimplemented!(),
        }
    }
}
