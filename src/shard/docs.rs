use super::content::{ContentIdx, ContentStore};
use derive_more::{Add, From, Into, Sub};
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::rc::Rc;
use sucds::elias_fano::EliasFano;

#[derive(Copy, Clone, Add, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash)]
pub struct DocID(u32);

pub struct DocStore {
    file: Rc<File>,
    content: ContentStore,
    doc_ends_ptr: u64,
    doc_ends_len: usize,
}

impl DocStore {
    pub fn new(
        doc_ends_ptr: u64,
        doc_ends_len: usize,
        file: Rc<File>,
        content: ContentStore,
    ) -> Self {
        Self {
            file,
            content,
            doc_ends_ptr,
            doc_ends_len,
        }
    }

    pub fn read_content(&self, doc_id: DocID, doc_ends: &DocEnds) -> Result<Vec<u8>, io::Error> {
        let range = doc_ends.content_range(doc_id);
        let mut buf = Vec::with_capacity(u32::from(range.end - range.start) as usize);
        (*self.file).read_exact_at(&mut buf, u64::from(range.start))?;
        Ok(buf)
    }

    // Returns the list of offsets (relative to the beginning of the content block)
    // that contain the zero-byte separators at the end of each document.
    pub fn read_doc_ends(&self) -> Result<DocEnds, io::Error> {
        let doc_ends = vec![ContentIdx(0u32); self.doc_ends_len as usize];
        let mut doc_ends_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                doc_ends.as_ptr() as *mut u8,
                doc_ends.len() * std::mem::size_of::<u32>(),
            )
        };
        (*self.file).read_exact_at(&mut doc_ends_bytes, self.doc_ends_ptr)?;
        Ok(DocEnds(doc_ends))
    }
}

pub struct CachedDocStore {
    doc_store: DocStore,
}

pub struct DocEnds(Vec<ContentIdx>);

impl DocEnds {
    pub fn new(v: Vec<ContentIdx>) -> Self {
        Self(v)
    }

    // Returns the id of the document that contains the given content offset.
    pub fn container(&self, offset: ContentIdx) -> DocID {
        let id = self.0.partition_point(|end| offset > *end);
        DocID(id as u32)
    }

    // Returns the range (relative to the beginning of the content block) that
    // represents the content of the document with the given id.
    pub fn content_range(&self, id: DocID) -> Range<ContentIdx> {
        if id == DocID(0) {
            ContentIdx(0)..self.0[0]
        } else {
            self.0[u32::from(id) as usize - 1] + ContentIdx(1)..self.0[u32::from(id) as usize]
        }
    }

    pub fn compress(&self) -> CompressedDocEnds {
        CompressedDocEnds(
            EliasFano::from_ints(
                &self
                    .0
                    .iter()
                    .map(|&idx| idx.0 as usize)
                    .collect::<Vec<usize>>(),
            )
            .unwrap()
            .enable_rank(),
        )
    }
}

pub struct CompressedDocEnds(EliasFano);

impl CompressedDocEnds {
    pub fn new(e: EliasFano) -> Self {
        Self(e)
    }

    // Returns the id of the document that contains the given content offset.
    pub fn container(&self, offset: ContentIdx) -> DocID {
        DocID(self.0.rank(u32::from(offset) as usize) as u32)
    }

    // Returns the range (relative to the beginning of the content block) that
    // represents the content of the document with the given id.
    pub fn content_range(&self, id: DocID) -> Range<ContentIdx> {
        let start = if id == DocID(0) {
            ContentIdx(0)
        } else {
            ContentIdx(self.0.select(u32::from(id) as usize - 1) as u32 + 1)
        };
        let end = ContentIdx(self.0.select(u32::from(id) as usize) as u32);
        start..end
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
        let compressed = doc_ends.compress();

        let tests = vec![
            (ContentIdx(0), DocID(0)),
            (ContentIdx(1), DocID(0)),
            (ContentIdx(2), DocID(1)),
            (ContentIdx(11), DocID(3)),
            (ContentIdx(12), DocID(4)),
            (ContentIdx(99), DocID(6)),
            (ContentIdx(100), DocID(6)),
        ];

        for (content_idx, doc_id) in tests {
            assert_eq!(doc_ends.container(content_idx), doc_id);
            assert_eq!(compressed.container(content_idx), doc_id);
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
        let compressed = doc_ends.compress();

        let tests = vec![
            (DocID(0), ContentIdx(0)..ContentIdx(1)),
            (DocID(1), ContentIdx(2)..ContentIdx(5)),
            (DocID(2), ContentIdx(6)..ContentIdx(10)),
            (DocID(3), ContentIdx(11)..ContentIdx(11)),
            (DocID(4), ContentIdx(12)..ContentIdx(12)),
            (DocID(5), ContentIdx(13)..ContentIdx(20)),
            (DocID(6), ContentIdx(21)..ContentIdx(100)),
        ];

        for (doc_id, expected_range) in tests {
            assert_eq!(doc_ends.content_range(doc_id), expected_range);
            assert_eq!(compressed.content_range(doc_id), expected_range);
        }
    }
}
