use anyhow::Error;
use derive_more::{Add, AddAssign, Div, From, Into, Mul, Sub};
use std::ops::Range;
use std::sync::Arc;

use super::docs::ContentIdx;
use super::file::ShardFile;

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
pub struct SuffixBlock(pub Box<[ContentIdx]>);

impl SuffixBlock {
    // 2048 is chosen so SIZE_BYTES is 8192, which is a pretty standard page size.
    pub const SIZE_SUFFIXES: usize = 2048;
    pub const SIZE_BYTES: usize = Self::SIZE_SUFFIXES * std::mem::size_of::<u32>();
}

#[derive(Clone)]
pub struct SuffixArrayStore {
    file: Arc<ShardFile>,
    // Pointer to the suffix array relative to the start of the file
    // Length in u32s, not bytes
    // TODO this should not be public
    pub sa_len: u32,
}

impl SuffixArrayStore {
    pub fn new(file: Arc<ShardFile>, sa_len: u32) -> Self {
        Self { file, sa_len }
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

    pub fn read_block(&self, block_id: SuffixBlockID) -> Result<SuffixBlock, Error> {
        self.file.read_suffix_block(block_id)
    }
}
