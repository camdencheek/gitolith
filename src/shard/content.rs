use derive_more::{Add, Constructor, From, Into, Sub};
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::rc::Rc;

#[derive(Copy, Clone, Add, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash)]
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

#[derive(Debug)]
pub struct ContentStore {
    file: Rc<File>,
    content_ptr: u64,
    content_len: u32,
}

impl ContentStore {
    pub fn new(file: Rc<File>, content_ptr: u64, content_len: u32) -> Self {
        Self {
            file,
            content_ptr,
            content_len,
        }
    }

    pub fn read(&self, range: Range<ContentIdx>) -> Result<Vec<u8>, io::Error> {
        // Calculate the absolute file offsets for the given range
        let abs_start = u64::from(range.start) + self.content_ptr;
        let abs_end = u64::from(range.end) + self.content_ptr;

        debug_assert!(abs_start <= self.content_ptr + self.content_len as u64);
        debug_assert!(abs_end <= self.content_ptr + self.content_len as u64);

        let mut buf = vec![0u8; (abs_end - abs_start) as usize];
        (*self.file).read_exact_at(&mut buf, abs_start)?;
        Ok(buf)
    }
}