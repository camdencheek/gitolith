use derive_more::{Add, AddAssign, Div, From, Into, Mul, Sub};
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::unix::prelude::FileExt;
use std::sync::Arc;

use super::content::ContentIdx;

#[derive(
    Copy, Div, Mul, AddAssign, Clone, Add, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash,
)]
pub struct SuffixIdx(pub u32);

#[derive(
    Copy, Div, Mul, Clone, Add, AddAssign, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash,
)]
pub struct SuffixBlockID(pub u32);

impl From<SuffixBlockID> for u64 {
    fn from(block_id: SuffixBlockID) -> Self {
        block_id.0 as u64
    }
}

#[derive(Debug)]
pub struct SuffixBlock(pub [ContentIdx; Self::SIZE_SUFFIXES]);

impl SuffixBlock {
    // 2048 is chosen so SIZE_BYTES is 8192, which is a pretty standard page size.
    pub const SIZE_SUFFIXES: usize = 2048;
    pub const SIZE_BYTES: usize = Self::SIZE_SUFFIXES * std::mem::size_of::<u32>();

    fn new() -> Box<Self> {
        Box::new(Self([ContentIdx(0); Self::SIZE_SUFFIXES]))
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // TODO guarantee this is actually safe. I _think_ a single-element tuple struct will
        // always have the same representation as its only element, but I'm not 100% sure.
        // Maybe #[repr(C)] on ContentIdx would make me feel better.
        unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr() as *mut u8, Self::SIZE_BYTES) }
    }
}

#[derive(Clone)]
pub struct SuffixArrayStore {
    file: Arc<File>,
    // Pointer to the suffix array relative to the start of the file
    sa_ptr: u64,
    // Length in u32s, not bytes
    // TODO this should not be public
    pub sa_len: u32,
}

impl SuffixArrayStore {
    pub fn new(file: Arc<File>, sa_ptr: u64, sa_len: u32) -> Self {
        assert!(sa_ptr % SuffixBlock::SIZE_BYTES as u64 == 0);

        Self {
            file,
            sa_ptr,
            sa_len,
        }
    }

    pub fn max_block_id(&self) -> SuffixBlockID {
        if self.sa_len % SuffixBlock::SIZE_SUFFIXES as u32 == 0 {
            SuffixBlockID(self.sa_len / SuffixBlock::SIZE_SUFFIXES as u32)
        } else {
            SuffixBlockID(self.sa_len / SuffixBlock::SIZE_SUFFIXES as u32 + 1)
        }
    }

    pub fn block_range(suffix_range: Range<SuffixIdx>) -> Range<(SuffixBlockID, usize)> {
        let start = Self::block_id_for_suffix(suffix_range.start);
        let end = if u32::from(suffix_range.end) % SuffixBlock::SIZE_SUFFIXES as u32 == 0 {
            let (id, _) = Self::block_id_for_suffix(suffix_range.end);
            (id, SuffixBlock::SIZE_SUFFIXES)
        } else {
            Self::block_id_for_suffix(suffix_range.end)
        };
        start..end
    }

    // Returns the block ID for the block that contains the given suffix
    pub fn block_id_for_suffix(suffix: SuffixIdx) -> (SuffixBlockID, usize) {
        let SuffixIdx(suffix) = suffix;
        (
            SuffixBlockID(suffix / SuffixBlock::SIZE_SUFFIXES as u32),
            suffix as usize % SuffixBlock::SIZE_SUFFIXES,
        )
    }

    pub fn read_block(&self, block_id: SuffixBlockID) -> Result<Box<SuffixBlock>, io::Error> {
        // TODO assert that the block ID is in range
        let abs_start = self.sa_ptr + block_id.0 as u64 * SuffixBlock::SIZE_BYTES as u64;
        let mut block = SuffixBlock::new();
        (*self.file).read_exact_at(block.as_bytes_mut(), abs_start)?;
        Ok(block)
    }
}
