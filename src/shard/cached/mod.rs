use std::{
    ops::{Range, RangeInclusive},
    sync::Arc,
};

use crate::{
    cache::{Cache, CacheKey, CacheValue},
    search::regex::ConcatLiteralSet,
    shard::docs::ContentIdx,
};

use super::{
    docs::{DocEnds, DocID, DocStore},
    suffix::{SuffixArrayStore, SuffixBlock, SuffixBlockID, SuffixIdx},
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
        CachedDocs::new(self.id, self.shard.docs().clone(), self.cache.clone())
    }

    pub fn suffixes(&self) -> CachedSuffixes {
        CachedSuffixes::new(
            self.id,
            self.shard.suffixes().clone(),
            self.shard.docs().clone(),
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

    pub fn get_content(&self, doc_id: DocID, doc_ends: &DocEnds) -> Arc<[u8]> {
        let key = CacheKey::DocContent(self.shard_id, doc_id);
        let value = if let Some(v) = self.cache.get(&key) {
            v.value().clone()
        } else {
            let v = CacheValue::DocContent(
                self.docs
                    .read_content(doc_id, doc_ends)
                    .expect("failed to read doc content")
                    .into(),
            );
            self.cache.insert(key, v.clone(), 0);
            v
        };

        match value {
            CacheValue::DocContent(dc) => dc,
            _ => unreachable!(),
        }
    }

    pub fn get_doc_ends(&self) -> Arc<DocEnds> {
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

    pub fn lookup_literal_range<T>(&self, prefix_range: RangeInclusive<T>) -> Range<SuffixIdx>
    where
        T: AsRef<[u8]> + Eq,
    {
        let (start, end) = prefix_range.into_inner();
        let start_bound = self.partition_by_literal(start, false);
        let end_bound = self.partition_by_literal(end, true);
        start_bound..end_bound
    }

    fn partition_by_literal<T>(&self, prefix: T, include_equal: bool) -> SuffixIdx
    where
        T: AsRef<[u8]>,
    {
        use std::cmp::Ordering::*;
        let doc_ends = self.docs.get_doc_ends();

        let pred = |content_idx| {
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
                match cmp_ci(content, left) {
                    Less => return true,
                    Greater => return false,
                    _ => {}
                }
            }
            return include_equal;
        };

        self.partition_by_content_idx(pred)
    }

    fn partition_by_content_idx<T>(&self, mut pred: T) -> SuffixIdx
    where
        T: FnMut(ContentIdx) -> bool,
    {
        // Hold on to the last block we fetched because we will hit the same
        // block many times in a row at the end of the lookup. Hitting the cache
        // is cheap, but not that cheap.
        //
        // TODO: It may make sense to hold on to the last two fetched blocks so
        // binary searches that hop between blocks don't need to repeatedly hit the
        // cache. I tried this, but in a couple of tests it didn't seem to make much
        // of a difference.
        let mut last_block: Option<(SuffixBlockID, Arc<SuffixBlock>)> = None;

        let suffix_pred = |suffix_idx| -> bool {
            let (block_id, offset) = Self::block_id_for_suffix(suffix_idx);
            let block = match last_block.take() {
                Some((id, block)) => {
                    if id == block_id {
                        block
                    } else {
                        self.read_block(block_id)
                    }
                }
                None => self.read_block(block_id),
            };
            let content_idx = block.0[offset];
            last_block = Some((block_id, block));
            pred(content_idx)
        };

        self.partition_by_suffix_idx(suffix_pred)
    }

    fn partition_by_suffix_idx<T>(&self, mut pred: T) -> SuffixIdx
    where
        T: FnMut(SuffixIdx) -> bool,
    {
        let (mut min, mut max) = (SuffixIdx(0), SuffixIdx(self.suffixes.sa_len));

        while min < max {
            let mid = SuffixIdx((u32::from(max) - u32::from(min)) / 2 + u32::from(min));
            if pred(mid) {
                min = mid + SuffixIdx(1)
            } else {
                max = mid
            }
        }
        min
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
}

fn cmp_ci(left: &[u8], right: &[u8]) -> std::cmp::Ordering {
    use std::cmp::Ordering::*;

    for (l, r) in left.iter().copied().zip(right.iter().copied()) {
        match l.to_ascii_lowercase().cmp(&r.to_ascii_lowercase()) {
            Less => return Less,
            Greater => return Greater,
            _ => {}
        }
    }

    left.len().cmp(&right.len())
}

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
    type Item = (Arc<[u8]>, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.range.end - self.range.start == ContentIdx(0) {
            return None;
        }

        let next_doc_id = self.doc_ends.find(self.range.start);
        let next_doc_range = self.doc_ends.content_range(next_doc_id);
        let doc_content = self.docs.get_content(next_doc_id, self.doc_ends);

        if next_doc_range.end < self.range.end {
            let res = Some((
                doc_content,
                usize::from(self.range.start - next_doc_range.start)
                    ..usize::from(next_doc_range.end - next_doc_range.start),
            ));
            self.range.start = next_doc_range.end;
            res
        } else {
            let res = Some((
                doc_content,
                usize::from(self.range.start - next_doc_range.start)
                    ..usize::from(self.range.end - next_doc_range.start),
            ));
            self.range.start = self.range.end;
            res
        }
    }
}

pub struct SuffixRangeIterator {
    states: <Range<usize> as IntoIterator>::IntoIter,
    range_set: ConcatLiteralSet,
    suffixes: CachedSuffixes,
    buf: Vec<u8>,
}

impl SuffixRangeIterator {
    pub fn new(range_set: ConcatLiteralSet, suffixes: CachedSuffixes) -> Self {
        Self {
            states: (0..range_set.cardinality()),
            range_set,
            suffixes,
            buf: Vec::new(),
        }
    }
}

impl Iterator for SuffixRangeIterator {
    type Item = (Range<SuffixIdx>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let state = self.states.next()?;
        self.buf.clear();
        self.range_set.write_nth_literal_to(state, &mut self.buf);

        Some((
            self.suffixes.lookup_literal_range(&self.buf..=&self.buf),
            self.buf.len(),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.states.len(), Some(self.states.len()))
    }
}

impl ExactSizeIterator for SuffixRangeIterator {
    fn len(&self) -> usize {
        self.states.len()
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
