use anyhow::Error;
use std::io::{self, Write};
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

struct ParallelDocChecker<T: Send + Sync> {
    doc_ids: T,
    docs: CachedDocs,
    doc_ends: Arc<DocEnds>,
    re: Regex,
    id_buffer: Vec<Option<DocID>>,
    result_buffer: Vec<Option<Result<DocMatch, Error>>>,
    result_buffer_next_idx: usize,
}

impl<T> ParallelDocChecker<T>
where
    T: Iterator<Item = DocID> + Send + Sync,
{
    // TODO tune this. Higher means a larger chunk in memory, but lower means less
    // flexibility in parallelism for work stealing. Right now, we do a chunk in parallel
    // at a time, but if we're willing to give up scoping of background tasks we could
    // kick off workers in the background, we can keep searching will processing a batch.
    // Realistically, I'm not sure how useful this is since we already get decent single-shard
    // performance, and we'll parallelize across shards anyways.
    const BUFFER_SIZE: usize = 256;

    fn new(doc_ids: T, docs: CachedDocs, doc_ends: Arc<DocEnds>, re: Regex) -> Self {
        Self {
            doc_ids,
            docs,
            doc_ends,
            re,
            id_buffer: vec![None; Self::BUFFER_SIZE],
            result_buffer: (0..Self::BUFFER_SIZE).map(|_| None).collect(),
            result_buffer_next_idx: Self::BUFFER_SIZE,
        }
    }

    fn fill_buffer(&mut self) {
        let mut id_buffer: Vec<Option<DocID>> = std::mem::take(&mut self.id_buffer);
        let mut result_buffer: Vec<Option<Result<DocMatch, Error>>> =
            std::mem::take(&mut self.result_buffer);

        id_buffer.fill(None);
        result_buffer.fill_with(|| None);

        self.fill_ids(&mut id_buffer);
        self.search_ids(&mut id_buffer, &mut result_buffer);

        self.id_buffer = id_buffer;
        self.result_buffer = result_buffer;
    }

    fn fill_ids(&mut self, ids: &mut [Option<DocID>]) {
        for id in ids.iter_mut() {
            *id = self.doc_ids.next();
        }
    }

    fn search_ids(
        &self,
        id_buf: &mut [Option<DocID>],
        result_buf: &mut [Option<Result<DocMatch, Error>>],
    ) {
        if id_buf.len() == 1 {
            if let Some(id) = id_buf[0] {
                result_buf[0] = Some(self.search_id(id))
            } else {
                result_buf[0] = None
            }
        } else {
            let mid = id_buf.len() / 2;
            let (id_buf_left, id_buf_right) = id_buf.split_at_mut(mid);
            let (result_buf_left, result_buf_right) = result_buf.split_at_mut(mid);
            rayon::join(
                || self.search_ids(id_buf_left, result_buf_left),
                || self.search_ids(id_buf_right, result_buf_right),
            );
        }
    }

    fn search_id(&self, doc_id: DocID) -> Result<DocMatch, Error> {
        let content = self.docs.read_content(doc_id, &self.doc_ends)?;
        let matched_ranges: Vec<Range<u32>> = self
            .re
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
    T: Iterator<Item = DocID> + Send + Sync,
{
    type Item = Result<DocMatch, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.result_buffer_next_idx >= self.result_buffer.len() {
                self.fill_buffer();
                self.result_buffer_next_idx = 0;
            }

            let doc_result = self.result_buffer[self.result_buffer_next_idx].take()?;
            self.result_buffer_next_idx += 1;
            match doc_result {
                Ok(doc_match) => {
                    if doc_match.matches.len() > 0 {
                        return Some(Ok(doc_match));
                    } else {
                        continue;
                    }
                }
                e @ Err(_) => return Some(e),
            }
        }
    }
}
