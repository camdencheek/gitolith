use super::file::ShardStore;
use anyhow::Error;
use derive_more::{Add, AddAssign, From, Into, Mul, Sub};
use std::ops::Range;
use std::sync::Arc;

#[derive(
    Copy, Clone, AddAssign, Add, Sub, PartialEq, Mul, From, Into, PartialOrd, Debug, Eq, Hash,
)]
pub struct DocID(pub u32);

// TODO create a macro that derives iterators for all the range types
#[derive(Copy, Clone, Add, Sub, PartialEq, Mul, From, Into, PartialOrd, Debug, Eq, Hash)]
#[mul(forward)]
pub struct DocOffset(pub u32);

impl From<DocID> for usize {
    fn from(doc_id: DocID) -> Self {
        doc_id.0 as usize
    }
}

impl From<DocID> for u64 {
    fn from(doc_id: DocID) -> Self {
        doc_id.0 as u64
    }
}

impl DocID {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn into_iter(range: Range<DocID>) -> impl ExactSizeIterator<Item = DocID> {
        (range.start.0 as u32..range.end.0 as u32).map(DocID)
    }
}

#[derive(
    Default, Copy, Clone, Add, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Ord, Hash,
)]
pub struct ContentIdx(pub u32);

impl From<ContentIdx> for usize {
    fn from(ci: ContentIdx) -> Self {
        u32::from(ci) as usize
    }
}

impl From<ContentIdx> for u64 {
    fn from(ci: ContentIdx) -> Self {
        u32::from(ci) as u64
    }
}

#[derive(Clone)]
pub struct DocStore {
    store: ShardStore,
}

impl DocStore {
    pub fn new(store: ShardStore) -> Self {
        Self { store }
    }

    pub fn doc_ids(&self) -> impl Iterator<Item = DocID> {
        (0..self.num_docs()).into_iter().map(DocID)
    }

    pub fn read_content(&self, doc_id: DocID, doc_ends: &DocEnds) -> Result<Arc<[u8]>, Error> {
        self.store.read_doc(doc_id, doc_ends)
    }

    // Returns the list of offsets (relative to the beginning of the content block)
    // that contain the zero-byte separators at the end of each document.
    pub fn read_doc_ends(&self) -> Result<Arc<DocEnds>, Error> {
        self.store.read_doc_ends()
    }

    pub fn num_docs(&self) -> u32 {
        (self.store.header().docs.offsets.len / std::mem::size_of::<u32>() as u64) as u32
    }

    pub fn max_doc_id(&self) -> DocID {
        DocID(self.num_docs() - 1)
    }
}

// TODO this can be slightly more efficient as a boxed slice
#[derive(Debug)]
pub struct DocEnds(pub Vec<ContentIdx>);

// TODO consider compressing doc ends with Elias Fano to minimize memory impact.
impl DocEnds {
    pub fn new(v: Vec<ContentIdx>) -> Self {
        Self(v)
    }

    pub fn iter_docs(&self) -> DocIDIterator {
        DocIDIterator::new(self.doc_count() as u32)
    }

    pub fn doc_count(&self) -> usize {
        self.0.len()
    }

    // Returns the id of the document that contains the given content offset.
    pub fn find(&self, offset: ContentIdx) -> DocID {
        DocID(self.0.partition_point(|&end| end <= offset) as u32)
    }

    // Returns the range (relative to the beginning of the content block) that
    // represents the content of the document with the given id.
    pub fn content_range(&self, id: DocID) -> Range<ContentIdx> {
        self.doc_start(id)..self.doc_end(id)
    }

    pub fn doc_start(&self, doc: DocID) -> ContentIdx {
        if doc == DocID(0) {
            ContentIdx(0)
        } else {
            self.0[usize::from(doc) - 1]
        }
    }

    pub fn doc_end(&self, doc: DocID) -> ContentIdx {
        self.0[usize::from(doc)]
    }

    pub fn max_doc_id(&self) -> DocID {
        DocID(self.0.len() as u32 - 1)
    }
}

pub struct DocIDIterator {
    next_doc: DocID,
    max_doc: DocID,
}

impl DocIDIterator {
    fn new(count: u32) -> Self {
        Self {
            next_doc: DocID(0),
            max_doc: DocID(count + 1),
        }
    }
}

impl Iterator for DocIDIterator {
    type Item = DocID;

    fn next(&mut self) -> Option<Self::Item> {
        let next_doc = self.next_doc;
        if next_doc >= self.max_doc {
            None
        } else {
            self.next_doc += DocID(1);
            Some(next_doc)
        }
    }
}

impl ExactSizeIterator for DocIDIterator {
    fn len(&self) -> usize {
        self.max_doc.0 as usize
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_doc_ends_container() {
        let doc_ends = DocEnds::new(vec![
            ContentIdx(1),
            ContentIdx(5),
            ContentIdx(10),
            ContentIdx(11),
            ContentIdx(12),
            ContentIdx(20),
            ContentIdx(100),
        ]);

        let tests = vec![
            (ContentIdx(0), DocID(0)),
            (ContentIdx(1), DocID(1)),
            (ContentIdx(2), DocID(1)),
            (ContentIdx(11), DocID(4)),
            (ContentIdx(12), DocID(5)),
            (ContentIdx(99), DocID(6)),
            (ContentIdx(100), DocID(7)),
        ];

        for (content_idx, doc_id) in tests {
            assert_eq!(doc_ends.find(content_idx), doc_id, "{:?}", content_idx);
            // assert_eq!(compressed.container(content_idx), doc_id);
        }
    }

    #[test]
    fn test_doc_ends_content_range() {
        let doc_ends = DocEnds::new(vec![
            ContentIdx(1),
            ContentIdx(5),
            ContentIdx(10),
            ContentIdx(11),
            ContentIdx(12),
            ContentIdx(20),
            ContentIdx(100),
        ]);

        let tests = vec![
            (DocID(0), ContentIdx(0)..ContentIdx(1)),
            (DocID(1), ContentIdx(1)..ContentIdx(5)),
            (DocID(2), ContentIdx(5)..ContentIdx(10)),
            (DocID(3), ContentIdx(10)..ContentIdx(11)),
            (DocID(4), ContentIdx(11)..ContentIdx(12)),
            (DocID(5), ContentIdx(12)..ContentIdx(20)),
            (DocID(6), ContentIdx(20)..ContentIdx(100)),
        ];

        for (doc_id, expected_range) in tests {
            assert_eq!(doc_ends.content_range(doc_id), expected_range);
        }
    }
}
