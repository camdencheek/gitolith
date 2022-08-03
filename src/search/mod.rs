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

pub fn search_regex(
    s: CachedShard,
    query: &str,
    skip_index: bool,
) -> Result<Box<dyn Iterator<Item = DocMatch>>, Error> {
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

    // TODO implement exact matching
    extracted = match extracted {
        ExtractedRegexLiterals::Exact(e) => ExtractedRegexLiterals::Inexact(vec![e]),
        _ => extracted,
    };

    match extracted {
        ExtractedRegexLiterals::None => {
            let doc_ends = s.docs().read_doc_ends();
            let suffixes = s.suffixes();
            let re = Arc::new(re);
            let doc_ids = s.docs().doc_ids();
            let docs = Arc::new(s.docs());
            Ok(Box::new(
                doc_ids
                    .par_map(128, move |doc_id| -> DocMatch {
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
                    .filter(|doc_match| doc_match.matches.len() > 0)
            ))
        }
        ExtractedRegexLiterals::Exact(set) => {
            todo!()
        }
        ExtractedRegexLiterals::Inexact(all) => {
            let doc_ends = s.docs().read_doc_ends();
            let suffixes1 = s.suffixes();
            let doc_ids = s.docs().doc_ids();
            let docs = Arc::new(s.docs());
            Ok(Box::new(
                    InexactDocIterator::new(
                        doc_ends.clone(),
                        all
                            .into_iter()
                            .map(move |rs| SuffixRangeIterator::new(rs, suffixes1.clone()))
                            .map(|suf_range_iter| {
                                let suffixes2 = s.suffixes();
                                suf_range_iter
                                    .map(move |suf_range| {
                                        ContentIdxIterator::new(suf_range, suffixes2.clone())
                                    })
                                    .flatten()
                            })
                            .collect()
                    ).par_map(64, move |doc_id| -> DocMatch {
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
            .filter(|doc_match| doc_match.matches.len() > 0)
                ))
        }
        // ExtractedRegexLiterals::Inexact(all) => Ok(Box::new(all.iter().map(
        //     |prefix_set| -> Result<BitVec, Error> {
        //         let mut start = Vec::new();
        //         let mut end = Vec::new();

        //         for i in prefix_set.len() {
        //             start.clear();
        //             end.clear();
        //             prefix_set.write_state_to(i, &mut start, &mut end);
        //         }
        //     },
        // ))),
    }
}

struct InexactDocIterator<T> {
    seen_docs: Vec<BitVec>,
    next_doc_id: DocID,
    max_doc_id: DocID,
    content_idx_iters: Vec<T>,
    doc_ends: Arc<DocEnds>,
}

impl<T> InexactDocIterator<T>
where
    T: Iterator<Item = ContentIdx>,
{
    fn new(doc_ends: Arc<DocEnds>, content_idx_iters: Vec<T>) -> Self {
        Self {
            seen_docs: vec![bitvec::bitvec![0; doc_ends.doc_count()]; content_idx_iters.len()],
            next_doc_id: DocID(0),
            max_doc_id: doc_ends.max_doc_id(),
            content_idx_iters,
            doc_ends,
        }
    }

    fn iter_contains_doc(&mut self, iter_idx: usize, doc_id: DocID) -> bool {
        let mut seen_docs = &mut self.seen_docs[iter_idx];

        if seen_docs[usize::from(doc_id)] {
            return true;
        }

        let mut idx_iter = &mut self.content_idx_iters[iter_idx];
        while let Some(content_idx) = idx_iter.next() {
            // TODO we can probably speed this up by hinting the starting range
            let idx_doc_id = self.doc_ends.find(content_idx);
            if idx_doc_id == doc_id {
                return true;
            } else if idx_doc_id > doc_id {
                seen_docs.set(usize::from(idx_doc_id), true);
            }
        }
        false
    }
}

impl<T> Iterator for InexactDocIterator<T>
where
    T: Iterator<Item = ContentIdx>,
{
    type Item = DocID;

    fn next(&mut self) -> Option<Self::Item> {
        'OUTER: while self.next_doc_id <= self.max_doc_id {
            let doc_id = self.next_doc_id;
            self.next_doc_id += DocID(1);

            for i in 0..self.content_idx_iters.len() {
                if !self.iter_contains_doc(i, doc_id) {
                    continue 'OUTER;
                }
            }

            return Some(doc_id);
        }

        return None;
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

struct ParallelMap<T, R, I, F>
where
    I: Iterator<Item = T>,
{
    f: F,
    input: I,
    senders: Vec<Sender<R>>,
    receivers: Vec<Receiver<R>>,
    next_receiver: Cycle<Range<usize>>,
}

impl<T, R, I, F> ParallelMap<T, R, I, F>
where
    F: Fn(T) -> R + Send + Sync + Clone + 'static,
    I: Iterator<Item = T>,
    T: Send + 'static,
    R: Send + 'static,
{
    fn new(input: I, max_parallel: usize, f: F) -> Self {
        Self {
            f,
            input,
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
        rayon::spawn_fifo(move || {
            let r = f(next_input);
            tx.send(r);
        });
    }
}

impl<T, R, I, F> Iterator for ParallelMap<T, R, I, F>
where
    F: Fn(T) -> R + Send + Sync + Clone + 'static,
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

trait IteratorExt<T, R, I, F>
where
    I: Iterator<Item = T>,
{
    // Guarantees the same order
    fn par_map(self, max_parallel: usize, f: F) -> ParallelMap<T, R, I, F>;
}

impl<T, R, I, F> IteratorExt<T, R, I, F> for I
where
    F: Fn(T) -> R + Send + Sync + Clone + 'static,
    I: Iterator<Item = T>,
    T: Send + 'static,
    R: Send + 'static,
{
    fn par_map(self, max_parallel: usize, f: F) -> ParallelMap<T, R, I, F> {
        ParallelMap::new(self, max_parallel, f)
    }
}
