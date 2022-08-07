use anyhow::Error;
use derive_more::{Add, AddAssign, Div, From, Into, Mul, Sub};
use std::fs::File;
use std::io::{self, Read};
use std::ops::{Range, RangeInclusive};
use std::os::unix::prelude::FileExt;
use std::sync::Arc;
use sucds::elias_fano::EliasFano;
use sucds::{EliasFanoBuilder, Searial};

use super::content::{ContentIdx, ContentStore};
use suffix;

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
    content: ContentStore,
    // Pointer to the suffix array relative to the start of the file
    sa_ptr: u64,
    // Length in u32s, not bytes
    // TODO this should not be public
    pub sa_len: u32,
    trigrams_ptr: u64,
    trigrams_len: u64,
}

impl SuffixArrayStore {
    pub fn new(
        file: Arc<File>,
        content: ContentStore,
        sa_ptr: u64,
        sa_len: u32,
        trigrams_ptr: u64,
        trigrams_len: u64,
    ) -> Self {
        assert!(sa_ptr % SuffixBlock::SIZE_BYTES as u64 == 0);

        Self {
            file,
            content,
            sa_ptr,
            sa_len,
            trigrams_ptr,
            trigrams_len,
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
            let (id, offset) = Self::block_id_for_suffix(suffix_range.end);
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

    pub fn read_trigram_pointers(
        &self,
    ) -> Result<CompressedTrigramPointers, ReadTrigramPointersError> {
        let mut buf = vec![0u8; self.trigrams_len as usize];
        self.file.read_exact_at(&mut buf, self.trigrams_ptr)?;
        CompressedTrigramPointers::deserialize_from(buf.as_slice())
    }
}

// A set of pointers into the suffix array to the end (exclusive) of the range
// of suffixes that start with that trigram.
pub struct TrigramPointers(Vec<SuffixIdx>);

impl TrigramPointers {
    // The number of unique trigrams given a 256-character alphabet (u8)
    const N_TRIGRAMS: usize = 256 * 256 * 256;

    pub fn from_content(content: &[u8]) -> Self {
        // Use SuffixIdx in the frequencies array so we can mutate in-place
        // to get the pointers without any unsafe.
        let mut frequencies = vec![SuffixIdx(0); Self::N_TRIGRAMS];

        for i in 0..content.len() {
            let suffix = &content[i..];
            let trigram_idx = match suffix {
                [a, b, c, ..] => u32::from_be_bytes([0, *a, *b, *c]),
                [a, b] => u32::from_be_bytes([0, *a, *b, 0]),
                [a] => u32::from_be_bytes([0, *a, 0, 0]),
                _ => u32::from_be_bytes([0, 0, 0, 0]),
            };
            frequencies[trigram_idx as usize] += SuffixIdx(1)
        }

        // Calculate cumulative frequencies
        frequencies.iter_mut().fold(SuffixIdx(0), |mut acc, freq| {
            // Store the previous accumulator in freq
            std::mem::swap(&mut acc, freq);
            acc + *freq
        });
        let pointers = frequencies;
        Self(pointers)
    }

    pub fn compress(self) -> CompressedTrigramPointers {
        CompressedTrigramPointers::new(self.0)
    }
}

#[derive(Debug)]
pub struct CompressedTrigramPointers(EliasFano);

impl CompressedTrigramPointers {
    pub fn new(pointers: Vec<SuffixIdx>) -> Self {
        let universe = u32::from(pointers[TrigramPointers::N_TRIGRAMS - 1]) as usize + 1;
        let mut builder = EliasFanoBuilder::new(universe, pointers.len()).unwrap();
        for idx in pointers {
            builder.push(u32::from(idx) as usize).unwrap();
        }
        CompressedTrigramPointers(builder.build())
    }

    // Returns a range of suffix indexes that are guaranteed to contain the
    // bounds of the given range.
    pub fn bounds<T>(&self, r: RangeInclusive<T>) -> Range<SuffixIdx>
    where
        T: AsRef<[u8]>,
    {
        self.lower_bound(r.start())..self.upper_bound(r.end())
    }

    pub fn selectivity<T>(&self, r: RangeInclusive<T>) -> f64
    where
        T: AsRef<[u8]>,
    {
        (u32::from(self.upper_bound(r.end())) - u32::from(self.lower_bound(r.start()))) as f64
            / self.0.universe() as f64
    }

    // Returns an inclusive lower bound on the suffixes with the prefix needle
    pub fn lower_bound<T>(&self, needle: T) -> SuffixIdx
    where
        T: AsRef<[u8]>,
    {
        let idx = match needle.as_ref() {
            [a, b, c, ..] => u32::from_be_bytes([0, *a, *b, *c]),
            [a, b] => u32::from_be_bytes([0, *a, *b, 0]),
            [a] => u32::from_be_bytes([0, *a, 0, 0]),
            [] => 0,
        };
        SuffixIdx(self.0.select(idx as usize) as u32)
    }

    // Returns an exclusive upper bound on the suffixes with the prefix needle
    pub fn upper_bound<T>(&self, needle: T) -> SuffixIdx
    where
        T: AsRef<[u8]>,
    {
        // TODO audit these saturating adds carefully at the boundary conditions
        let idx = match needle.as_ref() {
            [a, b, c, ..] => u32::from_be_bytes([0, *a, *b, *c]).saturating_add(1),
            [a, b] => u32::from_be_bytes([0, 0, *a, *b]).saturating_add(1) << 8,
            [a] => u32::from_be_bytes([0, 0, 0, *a]).saturating_add(1) << 16,
            [] => self.0.len() as u32,
        };
        SuffixIdx(self.0.select(idx as usize) as u32)
    }

    pub fn serialize_into<W: std::io::Write>(
        &self,
        writer: W,
    ) -> Result<usize, WriteTrigramPointersError> {
        Ok(self.0.serialize_into(writer)?)
    }

    pub fn deserialize_from<R: Read>(reader: R) -> Result<Self, ReadTrigramPointersError> {
        Ok(Self(EliasFano::deserialize_from(reader)?))
    }

    pub fn size_in_bytes(&self) -> usize {
        self.0.size_in_bytes()
    }
}

#[derive(thiserror::Error, Debug)]
#[error("failed to write trigram pointers")]
pub struct WriteTrigramPointersError {
    #[from]
    source: anyhow::Error,
}

#[derive(thiserror::Error, Debug)]
pub enum ReadTrigramPointersError {
    #[error("failed to read trigram pointers")]
    IO(#[from] io::Error),
    #[error("failed to deserialize trigram pointers")]
    Deserialize(#[from] anyhow::Error),
}

#[cfg(test)]
mod test {
    use std::io::Stderr;

    use super::*;

    fn concat_strs(input: &[&str]) -> Vec<u8> {
        let mut content = Vec::new();
        for s in input {
            content.extend_from_slice(s.as_bytes());
            content.push(0);
        }
        content
    }

    fn concat_bytes(input: &[&[u8]]) -> Vec<u8> {
        let mut content = Vec::new();
        for s in input {
            content.extend_from_slice(s);
            content.push(0);
        }
        content
    }

    fn new_trigrams(content: &[u8]) -> CompressedTrigramPointers {
        TrigramPointers::from_content(content).compress()
    }

    fn new_sa(content: &[u8]) -> Vec<u32> {
        let mut sa = vec![0u32; content.len()];
        let mut stypes = suffix::SuffixTypes::new(sa.len() as u32);
        let mut bins = suffix::Bins::new();
        suffix::sais(sa.as_mut(), &mut stypes, &mut bins, &suffix::Utf8(content));
        sa
    }

    #[test]
    fn test_compressed() {
        let content = concat_strs(&[
            "document dockument",
            "document mockument",
            "document lockument",
            "document stockmument",
            "document schmockmument",
        ]);
        let trigrams = new_trigrams(&content);
        let sa = new_sa(&content);
        let bounds = trigrams.bounds(b"doc"..=b"doc");
        assert_eq!(bounds.end - bounds.start, SuffixIdx(6));
        for idx in u32::from(bounds.start)..u32::from(bounds.end) {
            assert!(content[sa[idx as usize] as usize..].starts_with(b"doc"))
        }

        let bounds = trigrams.bounds(b"do"..=b"do");
        assert_eq!(bounds.end - bounds.start, SuffixIdx(6));
        for idx in u32::from(bounds.start)..u32::from(bounds.end) {
            assert!(content[sa[idx as usize] as usize..].starts_with(b"do"))
        }

        let bounds = trigrams.bounds(b"d"..=b"d");
        assert_eq!(bounds.end - bounds.start, SuffixIdx(6));
        for idx in u32::from(bounds.start)..u32::from(bounds.end) {
            assert!(content[sa[idx as usize] as usize..].starts_with(b"d"))
        }

        let bounds = trigrams.bounds(b"men"..=b"men");
        assert_eq!(bounds.end - bounds.start, SuffixIdx(10));
        for idx in u32::from(bounds.start)..u32::from(bounds.end) {
            assert!(content[sa[idx as usize] as usize..].starts_with(b"men"))
        }

        let bounds = trigrams.bounds(b"ent"..=b"ent");
        assert_eq!(bounds.end - bounds.start, SuffixIdx(10));
        for idx in u32::from(bounds.start)..u32::from(bounds.end) {
            assert!(content[sa[idx as usize] as usize..].starts_with(b"ent"))
        }

        let bounds = trigrams.bounds(b"nt"..=b"nt");
        assert_eq!(bounds.end - bounds.start, SuffixIdx(10));
        for idx in u32::from(bounds.start)..u32::from(bounds.end) {
            assert!(content[sa[idx as usize] as usize..].starts_with(b"nt"))
        }
    }
}
