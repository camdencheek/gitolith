use std::io::{self, Write};
use std::ops::Range;

use ::regex::bytes::Regex;

use crate::shard::{
    docs::{CompressedDocEnds, DocEnds, DocID, DocStore},
    Shard,
};

use self::regex::{extract_regex_literals, ExtractedRegexLiterals};

pub mod regex;

pub struct DocMatch {
    pub id: DocID,
    pub matches: Vec<Range<u32>>,
    pub content: Vec<u8>,
}

pub fn search_regex(
    s: &Shard,
    query: &str,
    skip_index: bool,
) -> Result<Box<dyn Iterator<Item = Result<DocMatch, io::Error>>>, Box<dyn std::error::Error>> {
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

    match extracted {
        ExtractedRegexLiterals::None => {
            let doc_ends = s.docs.read_doc_ends()?;
            let doc_ids = s.docs.doc_ids();
            return Ok(Box::new(SequentialDocChecker::new(
                doc_ids,
                s.docs.clone(),
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
    doc_store: DocStore,
    doc_ends: DocEnds,
    re: Regex,
}

impl<T> SequentialDocChecker<T> {
    fn new(doc_ids: T, doc_store: DocStore, doc_ends: DocEnds, re: Regex) -> Self {
        Self {
            doc_ids,
            doc_store,
            doc_ends,
            re,
        }
    }
}

impl<T> Iterator for SequentialDocChecker<T>
where
    T: Iterator<Item = DocID>,
{
    type Item = Result<DocMatch, io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let doc_id = self.doc_ids.next()?;
            let content = match self.doc_store.read_content(doc_id, &self.doc_ends) {
                Err(e) => return Some(Err(e)),
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
