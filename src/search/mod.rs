use anyhow::Error;
use crossbeam::channel::{bounded, Receiver, RecvError, Sender};
use std::io::{self, Write};
use std::iter::Cycle;
use std::ops::Range;
use std::sync::Arc;

use ::regex::bytes::Regex;

use crate::shard::cached::{CachedDocs, CachedShard};
use crate::shard::{
    docs::{CompressedDocEnds, DocEnds, DocID, DocStore},
    Shard,
};

use self::regex::{extract_regex_literals, ExtractedRegexLiterals};

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
    let doc_ends = s.docs().read_doc_ends()?;
    let doc_ids = s.docs().doc_ids();

    match extracted {
        ExtractedRegexLiterals::None => {
            return Ok(Box::new(ParallelDocChecker::new(
                doc_ids,
                s.docs(),
                doc_ends,
                re,
            )));
        }
        ExtractedRegexLiterals::Exact(set) => {
            todo!()
        }
        ExtractedRegexLiterals::Inexact(all) => {
            todo!()
        }
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
            let content = match self.docs.read_content(doc_id, &self.doc_ends) {
                Err(e) => return Some(Err(e.into())),
                Ok(content) => content,
            };

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

struct ParallelDocChecker<T> {
    doc_ids: T,
    docs: Arc<CachedDocs>,
    doc_ends: Arc<DocEnds>,
    re: Arc<Regex>,
    senders: Vec<Sender<Result<DocMatch, Error>>>,
    receivers: Vec<Receiver<Result<DocMatch, Error>>>,
    next_idx: Cycle<Range<usize>>,
}

impl<T> ParallelDocChecker<T>
where
    T: Iterator<Item = DocID>,
{
    const QUEUE_SIZE: usize = 128;

    fn new(doc_ids: T, docs: CachedDocs, doc_ends: Arc<DocEnds>, re: Regex) -> Self {
        Self {
            doc_ids,
            docs: Arc::new(docs),
            doc_ends,
            re: Arc::new(re),
            senders: Vec::new(),
            receivers: Vec::new(),
            next_idx: (0..Self::QUEUE_SIZE).cycle(),
        }
    }

    fn init(&mut self) {
        self.senders = Vec::with_capacity(Self::QUEUE_SIZE);
        self.receivers = Vec::with_capacity(Self::QUEUE_SIZE);
        for i in 0..Self::QUEUE_SIZE {
            let (tx, rx) = bounded(1);
            self.receivers.push(rx);
            self.senders.push(tx);
            self.spawn_task(i);
        }
    }

    fn spawn_task(&mut self, idx: usize) {
        let doc_id = match self.doc_ids.next() {
            Some(id) => id,
            None => {
                let (tx, rx) = bounded(1);
                self.senders[idx] = tx;
                return;
            }
        };

        let docs = self.docs.clone();
        let doc_ends = self.doc_ends.clone();
        let re = self.re.clone();
        let tx = self.senders[idx].clone();

        rayon::spawn_fifo(move || {
            let res = Self::search_doc(docs, doc_ends, doc_id, re);
            tx.send(res);
        });
    }

    fn search_doc(
        docs: Arc<CachedDocs>,
        doc_ends: Arc<DocEnds>,
        doc_id: DocID,
        re: Arc<Regex>,
    ) -> Result<DocMatch, Error> {
        let content = docs.read_content(doc_id, &doc_ends)?;
        let matched_ranges: Vec<Range<u32>> = re
            .find_iter(&content)
            .map(|m| m.start() as u32..m.end() as u32)
            .collect();

        Ok(DocMatch {
            id: doc_id,
            matches: matched_ranges,
            content,
        })
    }
}

impl<T> Iterator for ParallelDocChecker<T>
where
    T: Iterator<Item = DocID>,
{
    type Item = Result<DocMatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.senders.len() == 0 {
            self.init();
        }

        loop {
            let idx = self.next_idx.next().unwrap();
            let rx = self.receivers[idx].clone();
            match rx.recv() {
                Err(RecvError) => return None,
                Ok(e @ Err(_)) => return Some(e),
                Ok(Ok(doc_match)) => {
                    self.spawn_task(idx);
                    if doc_match.matches.len() > 0 {
                        return Some(Ok(doc_match));
                    }
                }
            }
        }
    }
}
