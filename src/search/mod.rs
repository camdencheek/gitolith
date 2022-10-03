use anyhow::Error;
use bitvec::{self, vec::BitVec};
use dpc_pariter::IteratorExt;
use radsort;
use std::iter::Peekable;
use std::ops::Range;
use std::sync::Arc;

use ::regex::bytes::Regex;

use crate::shard::docs::{ContentIdx, DocIDIterator, DocStore};
use crate::shard::docs::{DocEnds, DocID};
use crate::shard::suffix::{
    SuffixArrayStore, SuffixBlock, SuffixBlockID, SuffixBlockIterator, SuffixIdx,
    SuffixRangeIterator,
};
use crate::shard::Shard;

use self::optimize::OptimizedLiterals;
use self::regex::{extract_regex_literals, ConcatLiteralSet, ExtractedRegexLiterals};

mod optimize;
pub mod regex;

use optimize::optimize_extracted;

#[derive(Clone)]
pub struct DocMatch {
    pub id: DocID,
    pub matches: Vec<Range<u32>>,
    pub content: Arc<[u8]>,
}

pub fn search_regex<'a>(
    s: Shard,
    query: &'_ str,
    skip_index: bool,
) -> Result<Box<dyn Iterator<Item = DocMatch> + 'a>, Error> {
    let ast = regex_syntax::ast::parse::Parser::new().parse(query)?;
    let hir = regex_syntax::hir::translate::Translator::new().translate(query, &ast)?;

    let extracted = if skip_index {
        ExtractedRegexLiterals::None
    } else {
        extract_regex_literals(hir)
    };

    let extracted = extracted.to_lower_ascii();
    let optimized = optimize_extracted(extracted);
    match optimized {
        OptimizedLiterals::None => Ok(new_unindexed_match_iterator(Regex::new(query)?, s)),
        OptimizedLiterals::OrderedExact(set) => new_exact_match_iterator(query, s, set),
        OptimizedLiterals::Inexact(all) => {
            Ok(new_inexact_match_iterator(Regex::new(query)?, s, all))
        }
    }
}

fn new_exact_match_iterator<'a>(
    query: &str,
    shard: Shard,
    literals: Vec<ConcatLiteralSet>,
) -> Result<Box<dyn Iterator<Item = DocMatch> + 'a>, Error> {
    let suffixes = shard.suffixes();

    let mut all_content_indexes = Vec::with_capacity(literals.len());
    for concat in &literals {
        let suf_ranges = SuffixRangeIterator::new(concat.clone(), suffixes.clone())
            .filter(|(suf_range, _)| suf_range.start != suf_range.end)
            .collect::<Vec<_>>();
        let content_idx_count = suf_ranges
            .iter()
            .map(|(range, _)| u32::from(range.end - range.start) as usize)
            .sum();
        if content_idx_count > (1 << 18) {
            // If the number of candidate matches is very high, fall
            // back to inexact matching, which will require a regex recheck
            // but does not require collecting the candidate set in memory.
            // TODO tune this. Collecting the content indexes in memory could
            // cause an OOM. Consider allocating this vec in a mmap.
            return Ok(new_inexact_match_iterator(
                Regex::new(query)?,
                shard,
                literals,
            ));
        }
        let mut content_indexes: Vec<(ContentIdx, u32)> = Vec::with_capacity(content_idx_count);
        for (range, len) in suf_ranges.into_iter() {
            for content_idx in ContentIdxIterator::new(range, &suffixes) {
                content_indexes.push((content_idx, len as u32));
            }
        }
        all_content_indexes.push(SortingIterator::new(content_indexes));
    }

    Ok(Box::new(ExactDocIter::new(
        shard.docs(),
        all_content_indexes,
    )))
}

fn new_inexact_match_iterator<'a>(
    re: Regex,
    shard: Shard,
    literals: Vec<ConcatLiteralSet>,
) -> Box<dyn Iterator<Item = DocMatch> + 'a> {
    let doc_ends = shard.docs().read_doc_ends().unwrap();
    let suffixes = shard.suffixes();
    let docs = Arc::new(shard.docs());

    let content_idx_iters = literals
        .into_iter()
        .map(|rs| SuffixRangeIterator::new(rs, suffixes.clone()))
        .map(|suf_range_iter| {
            let suffixes = shard.suffixes();
            suf_range_iter
                .map(|(suf_range, _)| suf_range)
                .filter(|suf_range| suf_range.start != suf_range.end)
                .flat_map(move |suf_range| ContentIdxIterator::new(suf_range, &suffixes))
        })
        .map(|content_idx_iter| ContentIdxDocIterator::new(doc_ends.clone(), content_idx_iter))
        .collect();
    let filtered = AndDocIterator::new(content_idx_iters)
        .parallel_map(move |doc_id| -> DocMatch {
            let content = docs.read_content(doc_id, &doc_ends).unwrap();
            let matched_ranges: Vec<Range<u32>> = re
                .find_iter(&content)
                .map(|m| m.start() as u32..m.end() as u32)
                .collect();

            DocMatch {
                id: doc_id,
                matches: matched_ranges,
                content,
            }
        })
        .filter(|doc_match| !doc_match.matches.is_empty());
    Box::new(filtered)
}

fn new_unindexed_match_iterator<'a>(
    re: Regex,
    shard: Shard,
) -> Box<dyn Iterator<Item = DocMatch> + 'a> {
    let doc_ends = shard.docs().read_doc_ends().unwrap();
    let doc_ids = shard.docs().doc_ids();
    let re = Arc::new(re);
    let docs = Arc::new(shard.docs());
    Box::new(
        doc_ids
            .parallel_map(move |doc_id| -> DocMatch {
                let content = docs.read_content(doc_id, &doc_ends).unwrap();
                let matched_ranges: Vec<Range<u32>> = re
                    .find_iter(&content)
                    .map(|m| m.start() as u32..m.end() as u32)
                    .collect();

                DocMatch {
                    id: doc_id,
                    matches: matched_ranges,
                    content,
                }
            })
            .filter(|doc_match| !doc_match.matches.is_empty()),
    )
}

struct ConcatIterator {
    matched_indexes: Vec<Peekable<SortingIterator>>,
}

impl ConcatIterator {
    fn new(matched_indexes: Vec<SortingIterator>) -> Self {
        Self {
            matched_indexes: matched_indexes.into_iter().map(|v| v.peekable()).collect(),
        }
    }

    /// Returns the total length
    fn has_successor(&mut self, location: ContentIdx, len: u32, index: usize) -> (bool, u32) {
        if index == self.matched_indexes.len() - 1 {
            return (true, len);
        }

        while let Some((loc2, len2)) =
            self.matched_indexes[index + 1].next_if(|(loc2, _)| loc2 <= &location)
        {
            if loc2 == location + ContentIdx(len) {
                let (ok, l) = self.has_successor(loc2, len2, index + 1);
                if ok {
                    return (true, len + l);
                }
            }
        }

        (false, 0)
    }
}

impl Iterator for ConcatIterator {
    type Item = (ContentIdx, u32);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (start, len) = self.matched_indexes[0].next()?;
            let (ok, len) = self.has_successor(start, len, 0);
            if ok {
                return Some((start, len));
            } else {
                continue;
            }
        }
    }
}

pub struct SortingIterator {
    inner: Vec<(ContentIdx, u32)>,
    next_idx: usize,
    sort_end: usize,
}

impl SortingIterator {
    fn new(inner: Vec<(ContentIdx, u32)>) -> Self {
        Self {
            inner,
            sort_end: 0,
            next_idx: 0,
        }
    }

    fn sort_next_block(&mut self) {
        const BLOCK_SIZE: usize = 1 << 13; // 8192
        let block_start = self.sort_end;
        let block_end = (self.sort_end + BLOCK_SIZE).min(self.inner.len());
        if block_end == self.sort_end + BLOCK_SIZE {
            self.inner[block_start..].select_nth_unstable(BLOCK_SIZE);
        }
        radsort::sort_by_key(&mut self.inner[block_start..block_end], |(idx, _)| {
            u32::from(*idx)
        });
        self.sort_end = block_end;
    }
}

impl Iterator for SortingIterator {
    type Item = (ContentIdx, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_idx >= self.inner.len() {
            return None;
        } else if self.next_idx >= self.sort_end {
            self.sort_next_block()
        }
        let res = self.inner[self.next_idx];
        self.next_idx += 1;
        Some(res)
    }
}

struct ExactDocIter {
    candidates: Peekable<ConcatIterator>,
    doc_ends: Arc<DocEnds>,
    docs: DocStore,
}

impl ExactDocIter {
    fn new(docs: DocStore, matched_indexes: Vec<SortingIterator>) -> Self {
        Self {
            doc_ends: docs.read_doc_ends().unwrap(),
            docs,
            candidates: ConcatIterator::new(matched_indexes).peekable(),
        }
    }
}

impl Iterator for ExactDocIter {
    type Item = DocMatch;

    fn next(&mut self) -> Option<Self::Item> {
        let (first_index, first_len) = self.candidates.next()?;
        let doc_id = self.doc_ends.find(first_index);
        let doc_range = self.doc_ends.content_range(doc_id);

        let mut res = DocMatch {
            id: doc_id,
            matches: Vec::with_capacity(1),
            content: self.docs.read_content(doc_id, &self.doc_ends).unwrap(),
        };

        let mut add_match = |idx: ContentIdx, len: u32| {
            let start = u32::from(idx - doc_range.start);
            let end = start + len;
            // Filter out matches that cross doc boundaries
            if end <= u32::from(doc_range.end) {
                res.matches.push(start..end);
            }
        };
        add_match(first_index, first_len);

        while let Some((idx, len)) = self.candidates.next_if(|(idx, _)| *idx < doc_range.end) {
            add_match(idx, len)
        }

        Some(res)
    }
}

struct AndDocIterator<T> {
    children: Vec<T>,
}

impl<T> AndDocIterator<T> {
    fn new(children: Vec<T>) -> Self {
        Self { children }
    }
}

impl<T> Iterator for AndDocIterator<T>
where
    T: Iterator<Item = DocID>,
{
    type Item = DocID;

    fn next(&mut self) -> Option<DocID> {
        let mut min_doc = DocID(0);
        let mut children_with_min_doc = 0;
        let num_children = self.children.len();

        for child_idx in (0..num_children).cycle() {
            // SAFETY: child_idx will never be out of bounds because we're iterating
            // over indexes from 0..self.children.len()
            let child = unsafe { self.children.get_unchecked_mut(child_idx) };

            loop {
                let next_doc = child.next()?;
                if next_doc < min_doc {
                    continue;
                }

                if next_doc == min_doc {
                    children_with_min_doc += 1;
                } else if next_doc > min_doc {
                    min_doc = next_doc;
                    children_with_min_doc = 1;
                }
                if children_with_min_doc == num_children {
                    // All children have yielded the current doc
                    return Some(min_doc);
                }
                break;
            }
        }
        None
    }
}

// ContentIdxDocFilter filters an iterator of DocIDs to only the docs
// that contain one of the ContentIdx yielded by a ContentIdx iterator.
struct ContentIdxDocIterator<C> {
    seen_docs: BitVec,
    doc_iter: DocIDIterator,
    content_idx_iter: C,
    doc_ends: Arc<DocEnds>,
}

impl<C> ContentIdxDocIterator<C> {
    fn new(doc_ends: Arc<DocEnds>, content_idx_iter: C) -> Self {
        let doc_iter = doc_ends.iter_docs();
        Self {
            seen_docs: bitvec::bitvec![0; doc_iter.len()],
            doc_iter: doc_ends.iter_docs(),
            content_idx_iter,
            doc_ends,
        }
    }
}

impl<C> Iterator for ContentIdxDocIterator<C>
where
    C: Iterator<Item = ContentIdx>,
{
    type Item = DocID;

    fn next(&mut self) -> Option<Self::Item> {
        for doc_id in self.doc_iter.by_ref() {
            if self.seen_docs[usize::from(doc_id)] {
                return Some(doc_id);
            }

            for content_idx in self.content_idx_iter.by_ref() {
                let content_doc_id = self.doc_ends.find(content_idx);
                if content_doc_id == doc_id {
                    return Some(doc_id);
                } else {
                    self.seen_docs.set(usize::from(content_doc_id), true);
                }
            }
        }
        None
    }
}

struct ContentIdxIterator {
    block_range: Range<(SuffixBlockID, usize)>,
    block_iter: SuffixBlockIterator,

    current_block: Option<Arc<SuffixBlock>>,
    current_block_idx_iter: <Range<usize> as IntoIterator>::IntoIter,
}

impl ContentIdxIterator {
    fn new(suffix_range: Range<SuffixIdx>, suffixes: &SuffixArrayStore) -> Self {
        let block_range = SuffixArrayStore::block_range(suffix_range);
        Self {
            block_iter: suffixes.iter_blocks(Some(block_range.start.0..=block_range.end.0)),
            block_range,
            current_block: None,
            current_block_idx_iter: 0..0,
        }
    }

    fn next_block(&mut self) {
        let (id, block) = match self.block_iter.next() {
            Some(n) => n,
            None => {
                self.current_block = None;
                return;
            }
        };
        let idx_start = if id == self.block_range.start.0 {
            self.block_range.start.1
        } else {
            0
        };

        let idx_end = if id == self.block_range.end.0 {
            self.block_range.end.1
        } else {
            SuffixBlock::SIZE_SUFFIXES
        };

        self.current_block = Some(block);
        self.current_block_idx_iter = idx_start..idx_end;
    }
}

impl Iterator for ContentIdxIterator {
    type Item = ContentIdx;

    fn next(&mut self) -> Option<Self::Item> {
        let mut block = match self.current_block {
            Some(ref b) => b,
            None => {
                self.next_block();
                self.current_block.as_ref()?
            }
        };

        let idx = match self.current_block_idx_iter.next() {
            Some(i) => i,
            None => {
                self.next_block();
                block = self.current_block.as_ref()?;
                self.current_block_idx_iter.next()?
            }
        };

        Some(block.0[idx])
    }
}

#[cfg(test)]
mod test {
    use std::{path::Path, sync::Arc};

    use crate::{
        cache,
        shard::{
            builder::ShardBuilder, cached_file::CachedShardFile, file::ShardBackend, Shard, ShardID,
        },
    };

    use super::search_regex;

    fn build_shard() -> Shard {
        let mut b = ShardBuilder::new(&Path::new("/tmp/testshard1")).unwrap();
        for doc in &mut [
            "document1".to_string(),
            "document2".to_string(),
            "contains needle".to_string(),
            "contains needle and another needle".to_string(),
            "contains case sensitive nEeDlE".to_string(),
            "line1\nline2".to_string(),
        ] {
            b.add_doc(doc.as_bytes()).unwrap();
        }
        let shard_file = b.build().unwrap();
        let c = cache::new_cache(64 * 1024 * 1024);
        let cached_store = Arc::new(CachedShardFile::new(ShardID(0), c, shard_file));
        Shard::from_store(cached_store)
    }

    fn assert_count(s: Shard, re: &str, want_count: usize) {
        let got_count: usize = search_regex(s, re, false)
            .unwrap()
            .map(|doc_match| doc_match.matches.len())
            .sum();
        assert_eq!(
            want_count, got_count,
            "expected different count for re '{}'",
            re
        );
    }

    #[test]
    fn test_regex() {
        let shard = build_shard();
        assert_count(shard.clone(), r"doc", 2);
        assert_count(shard.clone(), r"another", 1);
        assert_count(shard.clone(), r"needle", 3);
        assert_count(shard.clone(), r"ne.*ed.*le", 2);
        assert_count(shard.clone(), r"ne.*?ed.*?le", 3);
        assert_count(shard.clone(), r"(?i)needle", 4);
        assert_count(shard.clone(), r"(?i)ne\w*ed\w*le", 4);
        assert_count(shard.clone(), r".*", 7);
        assert_count(shard.clone(), r"(?s).*", 6);
        assert_count(shard.clone(), r"\w+", 15);
        assert_count(shard.clone(), "(?i)contains case sensitive", 1);
    }
}
