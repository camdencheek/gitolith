use anyhow::Error;
use derive_more::{Add, AddAssign, Div, From, Into, Mul, Sub};
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;

use crate::search::regex::ConcatLiteralSet;

use super::docs::{ContentIdx, DocEnds, DocStore};
use super::file::ShardStore;

#[derive(
    Copy, Div, Mul, AddAssign, Clone, Add, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash,
)]
pub struct SuffixIdx(pub u32);

#[derive(
    Copy, Div, Mul, Clone, Add, AddAssign, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash,
)]
pub struct SuffixBlockID(pub u32);

impl From<SuffixBlockID> for u64 {
    fn from(block_id: SuffixBlockID) -> Self {
        block_id.0 as u64
    }
}

#[derive(Debug)]
pub struct SuffixBlock(pub [ContentIdx; Self::SIZE_SUFFIXES]);

impl SuffixBlock {
    // 2048 is chosen so SIZE_BYTES is 8192, which is a pretty standard page size.
    pub const SIZE_SUFFIXES: usize = 2048;
    pub const SIZE_BYTES: usize = Self::SIZE_SUFFIXES * std::mem::size_of::<u32>();
}

impl Default for SuffixBlock {
    fn default() -> Self {
        Self([ContentIdx(0); Self::SIZE_SUFFIXES])
    }
}

#[derive(Clone)]
pub struct SuffixArrayStore {
    store: ShardStore,
}

impl SuffixArrayStore {
    pub fn new(store: ShardStore) -> Self {
        Self { store }
    }

    fn sa_len(&self) -> u32 {
        debug_assert!(
            self.store.header().content.data.len * std::mem::size_of::<u32>() as u64
                == self.store.header().sa.len
        );
        self.store.header().content.data.len as u32
    }

    pub fn max_block_id(&self) -> SuffixBlockID {
        if self.sa_len() % SuffixBlock::SIZE_SUFFIXES as u32 == 0 {
            SuffixBlockID(self.sa_len() / SuffixBlock::SIZE_SUFFIXES as u32)
        } else {
            SuffixBlockID(self.sa_len() / SuffixBlock::SIZE_SUFFIXES as u32 + 1)
        }
    }

    pub fn block_range(suffix_range: Range<SuffixIdx>) -> Range<(SuffixBlockID, usize)> {
        let start = Self::block_id_for_suffix(suffix_range.start);
        let end = if u32::from(suffix_range.end) % SuffixBlock::SIZE_SUFFIXES as u32 == 0 {
            let (id, _) = Self::block_id_for_suffix(suffix_range.end);
            (id, SuffixBlock::SIZE_SUFFIXES)
        } else {
            Self::block_id_for_suffix(suffix_range.end)
        };
        start..end
    }

    // Returns the block ID for the block that contains the given suffix
    pub fn block_id_for_suffix(suffix: SuffixIdx) -> (SuffixBlockID, usize) {
        let SuffixIdx(suffix) = suffix;
        (
            SuffixBlockID(suffix / SuffixBlock::SIZE_SUFFIXES as u32),
            suffix as usize % SuffixBlock::SIZE_SUFFIXES,
        )
    }

    pub fn iter_blocks(&self, range: Option<RangeInclusive<SuffixBlockID>>) -> SuffixBlockIterator {
        SuffixBlockIterator::new(self.clone(), range)
    }

    pub fn read_block(&self, block_id: SuffixBlockID) -> Result<Arc<SuffixBlock>, Error> {
        self.store.get_suffix_block(block_id)
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
        let doc_ends = self.store.get_doc_ends().unwrap();
        let docs = DocStore::new(Arc::clone(&self.store));

        let pred = |content_idx| {
            let mut prefix_slice: &[u8] = prefix.as_ref();
            let contents = ContiguousContentIterator::new(
                &docs,
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
            include_equal
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
                        self.read_block(block_id).unwrap()
                    }
                }
                None => self.read_block(block_id).unwrap(),
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
        let (mut min, mut max) = (SuffixIdx(0), SuffixIdx(self.sa_len()));

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
}

pub struct SuffixBlockIterator {
    last_id: SuffixBlockID,
    next_id: SuffixBlockID,
    suffixes: SuffixArrayStore,
}

impl SuffixBlockIterator {
    fn new(suffixes: SuffixArrayStore, range: Option<RangeInclusive<SuffixBlockID>>) -> Self {
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
            Some((next_id, self.suffixes.read_block(next_id).unwrap()))
        }
    }
}

pub struct SuffixRangeIterator {
    states: <Range<usize> as IntoIterator>::IntoIter,
    range_set: ConcatLiteralSet,
    suffixes: SuffixArrayStore,
    buf: Vec<u8>,
}

impl SuffixRangeIterator {
    pub fn new(range_set: ConcatLiteralSet, suffixes: SuffixArrayStore) -> Self {
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

struct ContiguousContentIterator<'a, 'b> {
    docs: &'a DocStore,
    doc_ends: &'b Arc<DocEnds>,
    range: Range<ContentIdx>,
}

impl<'a, 'b> ContiguousContentIterator<'a, 'b> {
    pub fn new(docs: &'a DocStore, doc_ends: &'b Arc<DocEnds>, range: Range<ContentIdx>) -> Self {
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
        let doc_content = self.docs.read_content(next_doc_id, self.doc_ends).unwrap();

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
