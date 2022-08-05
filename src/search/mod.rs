use anyhow::Error;
use bitvec::{self, vec::BitVec};
use crossbeam::channel::{bounded, Receiver, RecvError, Sender};
use std::io::{self, Write};
use std::iter::Cycle;
use std::ops::Range;
use std::sync::Arc;

use ::regex::bytes::Regex;

use crate::shard::cached::{
    CachedDocs, CachedShard, CachedSuffixes, SuffixBlockIterator, SuffixRangeIterator,
};
use crate::shard::content::ContentIdx;
use crate::shard::docs::DocIDIterator;
use crate::shard::suffix::{SuffixBlock, SuffixBlockID, SuffixIdx};
use crate::shard::{
    docs::{CompressedDocEnds, DocEnds, DocID, DocStore},
    Shard,
};

use self::regex::{extract_regex_literals, ExtractedRegexLiterals, PrefixRangeSet};

pub mod regex;

#[derive(Clone)]
pub struct DocMatch {
    pub id: DocID,
    pub matches: Vec<Range<u32>>,
    pub content: Arc<Vec<u8>>,
}

pub fn search_regex<'a>(
    s: CachedShard,
    query: &'_ str,
    skip_index: bool,
    scope: &'a rayon::ScopeFifo,
) -> Result<Box<dyn Iterator<Item = DocMatch> + 'a>, Error> {
    let re = Regex::new(query)?;
    let ast = regex_syntax::ast::parse::Parser::new()
        .parse(re.as_str())
        .expect("regex str failed to parse as AST");
    let hir = regex_syntax::hir::translate::Translator::new()
        .translate(re.as_str(), &ast)
        .expect("regex str failed to parse for translator");

    let mut extracted = if skip_index {
        ExtractedRegexLiterals::None
    } else {
        extract_regex_literals(hir)
    };
    // TODO optimize extracted

    extracted = match extracted {
        ExtractedRegexLiterals::Exact(e) => ExtractedRegexLiterals::Inexact(vec![e]),
        _ => extracted,
    };
    // dbg!(&extracted);
    extracted = extracted.optimize();
    // dbg!(&extracted);

    match extracted {
        ExtractedRegexLiterals::None => {
            let doc_ends = s.docs().read_doc_ends();
            let suffixes = s.suffixes();
            let re = Arc::new(re);
            let doc_ids = s.docs().doc_ids();
            let docs = Arc::new(s.docs());
            Ok(Box::new(
                doc_ids
                    .par_map(scope, 128, move |doc_id| -> DocMatch {
                        let content = docs.read_content(doc_id, &doc_ends);
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
                    // TODO would a par_filter_map be more efficient here?
                    .filter(|doc_match| doc_match.matches.len() > 0),
            ))
        }
        ExtractedRegexLiterals::Exact(set) => {
            todo!()
        }
        ExtractedRegexLiterals::Inexact(all) => {
            let doc_ends = s.docs().read_doc_ends();
            let suffixes = s.suffixes();
            let doc_ids = s.docs().doc_ids();
            let docs = Arc::new(s.docs());

            let content_idx_iters = all
                .into_iter()
                .map(|rs| SuffixRangeIterator::new(rs, suffixes.clone()))
                .map(|suf_range_iter| {
                    let suffixes2 = s.suffixes();
                    suf_range_iter
                        .filter(|suf_range| suf_range.start != suf_range.end)
                        .map(move |suf_range| ContentIdxIterator::new(suf_range, suffixes2.clone()))
                        .flatten()
                })
                .map(|content_idx_iter| {
                    ContentIdxDocIterator::new(doc_ends.clone(), content_idx_iter)
                })
                .collect();
            let filtered = AndDocIterator::new(content_idx_iters)
                .par_map(scope, 32, move |doc_id| -> DocMatch {
                    let content = docs.read_content(doc_id, &doc_ends);
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
                .filter(|doc_match| doc_match.matches.len() > 0);
            Ok(Box::new(filtered))
        }
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
            let mut child = unsafe { self.children.get_unchecked_mut(child_idx) };

            loop {
                let next_doc = child.next()?;
                if next_doc == min_doc {
                    children_with_min_doc += 1;
                } else if next_doc > min_doc {
                    // The current child doesn't contain min_doc,
                    // so update min_doc and continue searching the other
                    // children for the new one.
                    min_doc = next_doc;
                    children_with_min_doc = 1;
                    if children_with_min_doc == num_children {
                        // All children have yielded the current doc
                        return Some(min_doc);
                    }
                    break;
                }
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
        while let Some(doc_id) = self.doc_iter.next() {
            if self.seen_docs[usize::from(doc_id)] {
                return Some(doc_id);
            }

            while let Some(content_idx) = self.content_idx_iter.next() {
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

    suffixes: CachedSuffixes,
}

impl ContentIdxIterator {
    fn new(suffix_range: Range<SuffixIdx>, suffixes: CachedSuffixes) -> Self {
        let block_range = CachedSuffixes::block_range(suffix_range);
        Self {
            block_iter: suffixes.iter_blocks(Some(block_range.start.0..=block_range.end.0)),
            block_range,
            current_block: None,
            current_block_idx_iter: (0..0).into_iter(),
            suffixes,
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
        self.current_block_idx_iter = (idx_start..idx_end).into_iter();
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

struct SequentialDocChecker<T> {
    doc_ids: T,
    docs: CachedDocs,
    doc_ends: DocEnds,
    re: Regex,
}

impl<T> SequentialDocChecker<T> {
    fn new(doc_ids: T, docs: CachedDocs, doc_ends: DocEnds, re: Regex) -> Self {
        Self {
            doc_ids,
            docs,
            doc_ends,
            re,
        }
    }
}

impl<T> Iterator for SequentialDocChecker<T>
where
    T: Iterator<Item = DocID>,
{
    type Item = Result<DocMatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        // Loop until we find a doc with matches
        loop {
            let doc_id = self.doc_ids.next()?;
            let content = self.docs.read_content(doc_id, &self.doc_ends);
            let matched_ranges: Vec<Range<u32>> = self
                .re
                .find_iter(&content)
                .map(|m| m.start() as u32..m.end() as u32)
                .collect();

            if matched_ranges.len() > 0 {
                return Some(Ok(DocMatch {
                    id: doc_id,
                    matches: matched_ranges,
                    content,
                }));
            }
        }
    }
}

struct ParallelMap<'a, 'scope, T, R, I, F>
where
    I: Iterator<Item = T>,
{
    f: F,
    input: I,
    scope: &'a rayon::ScopeFifo<'scope>,
    senders: Vec<Sender<R>>,
    receivers: Vec<Receiver<R>>,
    next_receiver: Cycle<Range<usize>>,
}

impl<'a, 'scope, T, R, I, F> ParallelMap<'a, 'scope, T, R, I, F>
where
    F: Fn(T) -> R + Send + Sync + Clone + 'scope,
    I: Iterator<Item = T>,
    T: Send + 'static,
    R: Send + 'static,
{
    fn new(input: I, scope: &'a rayon::ScopeFifo<'scope>, max_parallel: usize, f: F) -> Self {
        Self {
            f,
            input,
            scope,
            senders: Vec::with_capacity(max_parallel),
            receivers: Vec::with_capacity(max_parallel),
            next_receiver: (0..max_parallel).cycle(),
        }
    }

    fn start_once(&mut self) {
        if self.senders.len() != 0 {
            return;
        }

        for i in 0..self.senders.capacity() {
            let (tx, rx) = bounded(1);
            self.receivers.push(rx);
            self.senders.push(tx);
            self.spawn_task(i);
        }
    }

    fn spawn_task(&mut self, idx: usize) {
        let next_input = match self.input.next() {
            Some(n) => n,
            None => {
                // Kinda hacky: replace the sender at the current
                // index in order to drop the one that was previously
                // there, closing the channel.
                let (tx, _) = bounded(1);
                self.senders[idx] = tx;
                return;
            }
        };

        let tx = self.senders[idx].clone();
        let f = self.f.clone();
        // TODO it would be best to pass in a scope here so we can propagate panics.
        self.scope.spawn_fifo(move |_| {
            let r = f(next_input);
            tx.send(r);
        });
    }
}

impl<'a, 'scope, T, R, I, F> Iterator for ParallelMap<'a, 'scope, T, R, I, F>
where
    F: Fn(T) -> R + Send + Sync + Clone + 'scope,
    I: Iterator<Item = T>,
    T: Send + 'static,
    R: Send + 'static,
{
    type Item = R;

    fn next(&mut self) -> Option<Self::Item> {
        self.start_once();

        loop {
            let idx = self.next_receiver.next().unwrap();
            let rx = self.receivers[idx].clone();
            match rx.recv() {
                Err(RecvError) => return None,
                Ok(r) => {
                    self.spawn_task(idx);
                    return Some(r);
                }
            }
        }
    }
}

trait IteratorExt<'a, 'scope, T, R, I, F>
where
    I: Iterator<Item = T>,
{
    // Guarantees the same order
    fn par_map(
        self,
        scope: &'a rayon::ScopeFifo<'scope>,
        max_parallel: usize,
        f: F,
    ) -> ParallelMap<'a, 'scope, T, R, I, F>;
}

impl<'a, 'scope, T, R, I, F> IteratorExt<'a, 'scope, T, R, I, F> for I
where
    F: Fn(T) -> R + Send + Sync + Clone + 'scope,
    I: Iterator<Item = T>,
    T: Send + 'static,
    R: Send + 'static,
{
    fn par_map(
        self,
        scope: &'a rayon::ScopeFifo<'scope>,
        max_parallel: usize,
        f: F,
    ) -> ParallelMap<'a, 'scope, T, R, I, F> {
        ParallelMap::new(self, scope, max_parallel, f)
    }
}
