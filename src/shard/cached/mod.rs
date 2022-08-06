use std::{
    io,
    ops::{Range, RangeInclusive},
    sync::Arc,
};

use crate::{
    cache::{Cache, CacheKey, CacheValue},
    search::regex::PrefixRangeSet,
};

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
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::DocContent(Arc::new(
                self.docs
                    .read_content(doc_id, doc_ends)
                    .expect("failed to read doc content"),
            ));
            self.cache.insert(key, v.clone(), 0);
            v
        };

        match value {
            CacheValue::DocContent(dc) => dc,
            _ => unreachable!(),
        }
    }

    pub fn read_doc_ends(&self) -> Arc<DocEnds> {
        let key = CacheKey::DocEnds(self.shard_id);
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::DocEnds(Arc::new(
                self.docs.read_doc_ends().expect("failed to read doc ends"),
            ));
            self.cache.insert(key, v.clone(), 0);
            v
        };

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

#[derive(Clone)]
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

    pub fn max_block_id(&self) -> SuffixBlockID {
        self.suffixes.max_block_id()
    }

    pub fn iter_blocks(&self, range: Option<RangeInclusive<SuffixBlockID>>) -> SuffixBlockIterator {
        SuffixBlockIterator::new(self.clone(), range)
    }

    pub fn lookup_prefix_range<T>(
        &self,
        trigrams: &Arc<CompressedTrigramPointers>,
        prefix_range: RangeInclusive<T>,
    ) -> Range<SuffixIdx>
    where
        T: AsRef<[u8]> + Eq,
    {
        let start_bounds = trigrams.bounds(prefix_range.start()..=prefix_range.start());
        let end_bounds = if prefix_range.start() == prefix_range.end() {
            start_bounds.clone()
        } else {
            trigrams.bounds(prefix_range.end()..=prefix_range.end())
        };
        if start_bounds.start == end_bounds.end {
            // Return early if there are no trigrams that match
            return start_bounds.start..end_bounds.end;
        }

        let (start, end) = prefix_range.into_inner();
        let start_bound = self.lookup_prefix_start(start, start_bounds);
        let end_bound = self.lookup_prefix_end(&end, start_bound..trigrams.upper_bound(&end));
        start_bound..end_bound
    }

    fn lookup_prefix_start<T>(&self, prefix: T, bounds: Range<SuffixIdx>) -> SuffixIdx
    where
        T: AsRef<[u8]>,
    {
        let doc_ends = self.docs.read_doc_ends();
        let pred = |suffix_idx| {
            // TODO we can probably improve perf here by holding onto the last block or two
            let (block_id, offset) = Self::block_id_for_suffix(suffix_idx);
            let block = self.read_block(block_id);
            let content_idx = block.0[offset];

            let mut prefix_slice: &[u8] = prefix.as_ref();
            let contents = ContiguousContentIterator::new(
                &self.docs,
                &doc_ends,
                content_idx..content_idx + ContentIdx(prefix_slice.len() as u32),
            );

            for (doc_content, slicer) in contents {
                let content = &doc_content[slicer];
                let (left, right) = prefix_slice.split_at(content.len());
                prefix_slice = right;
                if content < left {
                    return true;
                }
            }
            // prefix is equal
            return false;
        };
        self.find_suffix_idx(pred, Some(bounds))
    }

    fn lookup_prefix_end<T>(&self, prefix: T, bounds: Range<SuffixIdx>) -> SuffixIdx
    where
        T: AsRef<[u8]>,
    {
        let doc_ends = self.docs.read_doc_ends();
        let pred = |suffix_idx| {
            // TODO we can probably improve perf here by holding onto the last block or two
            let (block_id, offset) = Self::block_id_for_suffix(suffix_idx);
            let block = self.read_block(block_id);
            let content_idx = block.0[offset];

            let mut prefix_slice: &[u8] = prefix.as_ref();
            let contents = ContiguousContentIterator::new(
                &self.docs,
                &doc_ends,
                content_idx..content_idx + ContentIdx(prefix_slice.len() as u32),
            );

            for (doc_content, slicer) in contents {
                let content = &doc_content[slicer];
                let (left, right) = prefix_slice.split_at(content.len());
                prefix_slice = right;
                if content < left {
                    return true;
                }
            }
            // prefix is equal
            return true;
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
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::SuffixBlock(Arc::from(
                self.suffixes
                    .read_block(block_id)
                    .expect("failed to read suffix block"),
            ));
            self.cache.insert(key, v.clone(), 0);
            v
        };

        match value {
            CacheValue::SuffixBlock(b) => b,
            _ => unreachable!(),
        }
    }

    pub fn read_trigram_pointers(&self) -> Arc<CompressedTrigramPointers> {
        let key = CacheKey::TrigramPointers(self.shard_id);
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::TrigramPointers(Arc::new(
                self.suffixes
                    .read_trigram_pointers()
                    .expect("failed to read trigram pointers"),
            ));
            self.cache.insert(key, v.clone(), 0);
            v
        };

        match value {
            CacheValue::TrigramPointers(tp) => tp,
            _ => unimplemented!(),
        }
    }
}

// TODO AsRef<> instead of lifetimes here?
struct ContiguousContentIterator<'a, 'b> {
    docs: &'a CachedDocs,
    doc_ends: &'b Arc<DocEnds>,
    range: Range<ContentIdx>,
}

impl<'a, 'b> ContiguousContentIterator<'a, 'b> {
    pub fn new(docs: &'a CachedDocs, doc_ends: &'b Arc<DocEnds>, range: Range<ContentIdx>) -> Self {
        Self {
            docs,
            doc_ends,
            range,
        }
    }
}

impl<'a, 'b> Iterator for ContiguousContentIterator<'a, 'b> {
    type Item = (Arc<Vec<u8>>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.end - self.range.start == ContentIdx(0) {
            return None;
        }

        let next_doc_id = self.doc_ends.find(self.range.start);
        let next_doc_range = self.doc_ends.content_range(next_doc_id);
        let doc_content = self.docs.read_content(next_doc_id, self.doc_ends);

        if next_doc_range.end < self.range.end {
            let res = Some((
                doc_content,
                usize::from(self.range.start) - usize::from(next_doc_range.start)
                    ..usize::from(next_doc_range.end) - usize::from(next_doc_range.start),
            ));
            self.range.start = next_doc_range.end;
            res
        } else {
            let res = Some((
                doc_content,
                usize::from(self.range.start) - usize::from(next_doc_range.start)
                    ..usize::from(self.range.end) - usize::from(next_doc_range.start),
            ));
            self.range.start = self.range.end;
            res
        }
    }
}

pub struct SuffixRangeIterator {
    states: <Range<usize> as IntoIterator>::IntoIter,
    range_set: PrefixRangeSet,
    suffixes: CachedSuffixes,
    trigrams: Arc<CompressedTrigramPointers>,
    start_buf: Vec<u8>,
    end_buf: Vec<u8>,
}

impl SuffixRangeIterator {
    pub fn new(range_set: PrefixRangeSet, suffixes: CachedSuffixes) -> Self {
        Self {
            states: (0..range_set.len()).into_iter(),
            range_set,
            trigrams: suffixes.read_trigram_pointers(),
            suffixes,
            start_buf: Vec::new(),
            end_buf: Vec::new(),
        }
    }
}

impl Iterator for SuffixRangeIterator {
    type Item = Range<SuffixIdx>;

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.states.next()?;
        self.start_buf.clear();
        self.end_buf.clear();
        self.range_set
            .write_state_to(state, &mut self.start_buf, &mut self.end_buf);

        Some(
            self.suffixes
                .lookup_prefix_range(&self.trigrams, &self.start_buf..=&self.end_buf),
        )
    }
}

pub struct SuffixBlockIterator {
    last_id: SuffixBlockID,
    next_id: SuffixBlockID,
    suffixes: CachedSuffixes,
}

impl SuffixBlockIterator {
    fn new(suffixes: CachedSuffixes, range: Option<RangeInclusive<SuffixBlockID>>) -> Self {
        let range = match range {
            Some(r) => r,
            None => SuffixBlockID(0)..=suffixes.max_block_id(),
        };
        Self {
            suffixes,
            next_id: *range.start(),
            last_id: *range.end(),
        }
    }
}

impl Iterator for SuffixBlockIterator {
    type Item = (SuffixBlockID, Arc<SuffixBlock>);

    fn next(&mut self) -> Option<Self::Item> {
        let next_id = self.next_id;
        self.next_id += SuffixBlockID(1);
        if next_id > self.last_id {
            None
        } else {
            Some((next_id, self.suffixes.read_block(next_id)))
        }
    }
}
