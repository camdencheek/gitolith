use std::ops::{Index, Range, RangeFrom, RangeTo};

pub type DocID = u32;

#[derive(Copy, Clone)]
pub struct Doc<'a> {
    pub id: DocID,
    pub content: &'a [u8],
}

#[derive(Copy, Clone)]
pub struct Docs<'a> {
    start_id: DocID,
    start_offsets: &'a [u32],
    content: &'a [u8],
}

impl<'a> Docs<'a> {
    pub fn new(start_id: DocID, start_offsets: &'a [u32], content: &'a [u8]) -> Docs<'a> {
        Docs {
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
        I: DocsIndex<'a>,
    {
        idx.index(self)
    }

    pub fn pop_front(&mut self) -> Option<Doc<'a>> {
        if self.len() == 0 {
            None
        } else {
            let elem = self.index(0);
            *self = self.index(1..);
            Some(elem)
        }
    }

    // fn find_suffix(&self, suffix: u32) -> DocID {
    //     let (mut low, mut high) = (0usize, starts.len());
    //     while low != high {
    //         let mid = (low + high) / 2;
    //         if self.doc_start(mid as u32) > suffix {
    //             high = mid - 1
    //         } else if self.doc_end(mid as u32) < suffix {
    //             low = mid + 1
    //         } else {
    //             return mid as u32;
    //         }
    //     }
    // }
}

impl<'a> IntoIterator for Docs<'a> {
    type Item = Doc<'a>;
    type IntoIter = DocsIterator<'a>;

    fn into_iter(self) -> DocsIterator<'a> {
        DocsIterator { docs: self }
    }
}

pub struct DocsIterator<'a> {
    docs: Docs<'a>,
}

impl<'a> Iterator for DocsIterator<'a> {
    type Item = Doc<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.docs.pop_front()
    }
}

pub trait DocsIndex<'a> {
    type Output;

    fn index(&self, docs: &Docs<'a>) -> Self::Output;
}

impl<'a> DocsIndex<'a> for Range<u32> {
    type Output = Docs<'a>;

    fn index(&self, docs: &Docs<'a>) -> Docs<'a> {
        let content_start = docs.start_offsets[self.start as usize] as usize;
        let content_end = match docs.start_offsets.get(self.end as usize) {
            Some(off) => *off as usize,
            None => docs.content.len(),
        };

        Docs {
            start_id: docs.start_id + self.start,
            start_offsets: &docs.start_offsets[self.start as usize..self.end as usize],
            content: &docs.content[content_start..content_end],
        }
    }
}

impl<'a> DocsIndex<'a> for RangeFrom<u32> {
    type Output = Docs<'a>;

    fn index(&self, docs: &Docs<'a>) -> Docs<'a> {
        docs.index(self.start..docs.len())
    }
}

impl<'a> DocsIndex<'a> for RangeTo<u32> {
    type Output = Docs<'a>;

    fn index(&self, docs: &Docs<'a>) -> Docs<'a> {
        docs.index(0..self.end)
    }
}

impl<'a> DocsIndex<'a> for u32 {
    type Output = Doc<'a>;

    fn index(&self, docs: &Docs<'a>) -> Doc<'a> {
        let content_start = docs.start_offsets[*self as usize] as usize;
        let content_end = match docs.start_offsets.get(*self as usize + 1) {
            Some(off) => *off as usize - 1,
            None => docs.content.len() - 1,
        };
        Doc {
            id: docs.start_id + *self,
            content: &docs.content[content_start..content_end],
        }
    }
}