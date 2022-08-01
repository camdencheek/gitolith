use derive_more::{Add, From, Into, Sub};
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::sync::Arc;

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

#[derive(Debug, Clone)]
pub struct ContentStore {
    file: Arc<File>,
    file_ptr: u64,
    len: u32,
}

impl ContentStore {
    pub fn new(file: Arc<File>, content_ptr: u64, content_len: u32) -> Self {
        Self {
            file,
            file_ptr: content_ptr,
            len: content_len,
        }
    }

    pub fn read(&self, range: Range<ContentIdx>) -> Result<Vec<u8>, io::Error> {
        // No need to hit the disk for empty files
        if range.start == range.end {
            return Ok(Vec::new());
        }

        // Calculate the absolute file offsets for the given range
        let abs_start = u64::from(range.start) + self.file_ptr;
        let abs_end = u64::from(range.end) + self.file_ptr;

        debug_assert!(abs_start <= self.file_ptr + self.len as u64);
        debug_assert!(abs_end <= self.file_ptr + self.len as u64);

        let mut buf = vec![0u8; (abs_end - abs_start) as usize];
        (*self.file).read_exact_at(&mut buf, abs_start)?;
        Ok(buf)
    }
}
