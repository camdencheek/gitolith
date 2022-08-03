use anyhow::Error;
use bitvec::{self, vec::BitVec};
use crossbeam::channel::{bounded, Receiver, RecvError, Sender};
use std::io::{self, Write};
use std::iter::Cycle;
use std::ops::Range;
use std::sync::Arc;

use ::regex::bytes::Regex;

use crate::shard::cached::{CachedDocs, CachedShard, CachedSuffixes};
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
    s: &CachedShard,
    query: &str,
    skip_index: bool,
) -> Result<Box<dyn Iterator<Item = Result<DocMatch, Error>>>, Error> {
    let re = Regex::new(query)?;
    let ast = regex_syntax::ast::parse::Parser::new()
        .parse(re.as_str())
        .expect("regex str failed to parse as AST");
    let hir = regex_syntax::hir::translate::Translator::new()
        .translate(re.as_str(), &ast)
        .expect("regex str failed to parse for translator");

    let extracted = if skip_index {
        ExtractedRegexLiterals::None
    } else {
        extract_regex_literals(hir)
    };
    // TODO optimize extracted
    //
    let doc_ends = s.docs().read_doc_ends();
    let doc_ids = s.docs().doc_ids();
    let suffixes = s.suffixes();

    match extracted {
        ExtractedRegexLiterals::None => {
            let docs = Arc::new(s.docs());
            let re = Arc::new(re);
            let doc_ends = Arc::new(doc_ends);
            Ok(Box::new(
                doc_ids
                    .par_map(128, move |doc_id| -> Result<DocMatch, Error> {
                        let content = docs.read_content(doc_id, &doc_ends);
                        let matched_ranges: Vec<Range<u32>> = re
                            .find_iter(&content)
                            .map(|m| m.start() as u32..m.end() as u32)
                            .collect();

                        Ok(DocMatch {
                            id: doc_id,
                            matches: matched_ranges,
                            content,
                        })
                    })
                    // TODO would a par_filter_map be more efficient here?
                    .filter(|res| match res {
                        Ok(doc_match) => doc_match.matches.len() > 0,
                        Err(_) => true,
                    }),
            ))
        }
        ExtractedRegexLiterals::Exact(set) => {
            todo!()
        }
        ExtractedRegexLiterals::Inexact(all) => {
            todo!()
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

struct ContentIdxIterator {
    block_range: Range<(SuffixBlockID, usize)>,
    next_block_id: SuffixBlockID,
    suffixes: CachedSuffixes,
    current_block: Option<Arc<SuffixBlock>>,
}

impl ContentIdxIterator {
    fn new(suffix_range: Range<SuffixIdx>, suffixes: CachedSuffixes) -> Self {
        let block_range = CachedSuffixes::block_range(suffix_range);
        Self {
            next_block_id: block_range.start.0,
            block_range,
            suffixes,
            current_block: None,
        }
    }

    fn len_total(&self) -> usize {
        u32::from(self.block_range.end.0 - self.block_range.start.0) as usize
            * SuffixBlock::SIZE_SUFFIXES
            + (self.block_range.end.1 - self.block_range.start.1)
    }

    fn next(&mut self) -> Option<&[ContentIdx]> {
        if self.next_block_id > self.block_range.end.0 {
            return None;
        }

        let block_id = self.next_block_id;
        self.next_block_id += SuffixBlockID(1);
        self.current_block = Some(self.suffixes.read_block(self.next_block_id));

        let is_first_block = block_id == self.block_range.start.0;
        let is_last_block = block_id == self.block_range.end.0;

        let block_ref = &self.current_block.as_ref().unwrap().0;
        let slice = match (is_first_block, is_last_block) {
            (true, true) => &block_ref[self.block_range.start.1..self.block_range.end.1],
            (true, false) => &block_ref[self.block_range.start.1..],
            (false, true) => &block_ref[..self.block_range.end.1],
            (false, false) => &block_ref[..],
        };
        Some(slice)
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
