use super::suffix::SuffixIdx;
use std::ops::{Index, Range, RangeFrom, RangeTo};

pub type DocID = u32;

/// Doc is a view of a document in the index
#[derive(Copy, Clone)]
pub struct Doc<'s> {
    /// The index ID of the document. This is only guaranteed
    /// to be unique within the shard
    pub id: DocID,

    content_offset: u32,
    pub content: &'s [u8],
}

impl<'s> Doc<'s> {
    pub fn start(&self) -> u32 {
        self.content_offset
    }

    pub fn end(&self) -> u32 {
        self.content_offset + self.content.len() as u32
    }
}

#[derive(Copy, Clone)]
pub struct DocSlice<'s> {
    start_id: DocID,
    start_offsets: &'s [u32],
    content: &'s [u8],
}

impl<'s> DocSlice<'s> {
    pub fn new(start_id: DocID, start_offsets: &'s [u32], content: &'s [u8]) -> DocSlice<'s> {
        DocSlice {
            start_id,
            start_offsets,
            content,
        }
    }

    pub fn len(&self) -> u32 {
        self.start_offsets.len() as u32
    }

    pub fn index<I>(&self, idx: I) -> I::Output
    where
        I: DocsIndex<'s>,
    {
        idx.index(self)
    }

    pub fn pop_front(&mut self) -> Option<Doc<'s>> {
        if self.len() == 0 {
            None
        } else {
            let elem = self.index(0);
            *self = self.index(1..);
            Some(elem)
        }
    }

    pub fn peek(&self) -> Option<Doc<'s>> {
        if self.len() == 0 {
            None
        } else {
            Some(self.index(0))
        }
    }

    fn content_start(&self) -> u32 {
        self.start_offsets[0]
    }

    pub fn find_by_suffix(&self, suffix: SuffixIdx) -> Option<Doc<'s>> {
        // Check that the suffix is in bounds for this doc slice.
        if self.len() == 0 {
            return None;
        } else if self.content_start() > suffix {
            return None;
        } else if (self.content_start() + self.content.len() as u32) < suffix {
            return None;
        }

        let (mut low, mut high) = (0u32, self.len());
        while low <= high {
            let mid = (high - low) / 2 + low;
            let doc = self.index(mid);
            if doc.start() > suffix {
                high = mid - 1;
            } else if doc.end() < suffix {
                low = mid + 1
            } else {
                return Some(doc);
            }
        }
        None
    }
}

impl<'s> Iterator for DocSlice<'s> {
    type Item = Doc<'s>;

    fn next(&mut self) -> Option<Self::Item> {
        self.pop_front()
    }
}

pub trait DocsIndex<'a> {
    type Output;

    fn index(&self, docs: &DocSlice<'a>) -> Self::Output;
}

impl<'a> DocsIndex<'a> for Range<u32> {
    type Output = DocSlice<'a>;

    fn index(&self, docs: &DocSlice<'a>) -> DocSlice<'a> {
        let content_start = match docs.start_offsets.get(self.start as usize) {
            Some(off) => *off as usize - docs.start_offsets[0] as usize,
            None => docs.content.len(),
        };
        let content_end = match docs.start_offsets.get(self.end as usize) {
            Some(off) => *off as usize - docs.start_offsets[0] as usize,
            None => docs.content.len(),
        };

        DocSlice {
            start_id: docs.start_id + self.start,
            start_offsets: &docs.start_offsets[self.start as usize..self.end as usize],
            content: &docs.content[content_start..content_end],
        }
    }
}

impl<'a> DocsIndex<'a> for RangeFrom<u32> {
    type Output = DocSlice<'a>;

    fn index(&self, docs: &DocSlice<'a>) -> DocSlice<'a> {
        docs.index(self.start..docs.len())
    }
}

impl<'a> DocsIndex<'a> for RangeTo<u32> {
    type Output = DocSlice<'a>;

    fn index(&self, docs: &DocSlice<'a>) -> DocSlice<'a> {
        docs.index(0..self.end)
    }
}

impl<'a> DocsIndex<'a> for u32 {
    type Output = Doc<'a>;

    fn index(&self, docs: &DocSlice<'a>) -> Doc<'a> {
        let content_start =
            docs.start_offsets[*self as usize] as usize - docs.content_start() as usize;
        let content_end = match docs.start_offsets.get(*self as usize + 1) {
            Some(off) => *off as usize - 1 - docs.content_start() as usize,
            None => docs.content.len() - 1,
        };
        Doc {
            id: docs.start_id + *self,
            content_offset: docs.start_offsets[*self as usize],
            content: &docs.content[content_start..content_end],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
