use derive_more::{Add, AddAssign, From, Into, Sub};
use std::fs::File;
use std::io;
use std::ops::{Range, RangeInclusive};
use std::os::unix::prelude::FileExt;
use std::rc::Rc;
use sucds::elias_fano::EliasFano;
use sucds::EliasFanoBuilder;

#[derive(Copy, AddAssign, Clone, Add, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash)]
pub struct SuffixIdx(pub u32);

#[derive(Copy, Clone, Add, Sub, PartialEq, From, Into, PartialOrd, Debug, Eq, Hash)]
pub struct SuffixBlockID(pub u32);

pub struct SuffixBlock([u32; Self::SIZE_SUFFIXES]);

impl SuffixBlock {
    const SIZE_SUFFIXES: usize = 2048;
    const SIZE_BYTES: usize = Self::SIZE_SUFFIXES * std::mem::size_of::<u32>();

    fn new() -> Self {
        Self([0u32; Self::SIZE_SUFFIXES])
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr() as *mut u8, Self::SIZE_BYTES) }
    }
}

pub struct SuffixArrayStore {
    file: Rc<File>,
    // Pointer to the suffix array relative to the start of the file
    sa_ptr: u64,
    // Length in u32s, not bytes
    sa_len: u32,
}

impl SuffixArrayStore {
    const BLOCK_SIZE: usize = 8192;

    fn new(file: Rc<File>, sa_ptr: u64, sa_len: u32) -> Self {
        assert!(sa_ptr % SuffixBlock::SIZE_BYTES as u64 == 0);

        Self {
            file,
            sa_ptr,
            sa_len,
        }
    }

    // Returns the block ID for the block that contains the given suffix
    fn block_id_for_suffix(suffix: SuffixIdx) -> SuffixBlockID {
        SuffixBlockID(u32::from(suffix) / SuffixBlock::SIZE_SUFFIXES as u32)
    }

    fn read_block(&self, block_id: SuffixBlockID) -> Result<SuffixBlock, io::Error> {
        let abs_start = self.sa_ptr + block_id.0 as u64 * SuffixBlock::SIZE_BYTES as u64;
        let mut block = SuffixBlock::new();
        (*self.file).read_exact_at(block.as_bytes_mut(), abs_start)?;
        Ok(block)
    }
}

// A set of pointers into the suffix array to the end (exclusive) of the range
// of suffixes that start with that trigram.
struct TrigramPointers([SuffixIdx; Self::N_TRIGRAMS]);

impl TrigramPointers {
    // The number of unique trigrams given a 256-character alphabet (u8)
    const N_TRIGRAMS: usize = 256 * 256 * 256;

    pub fn from_content(content: &[u8]) -> Self {
        // Use SuffixIdx in the frequencies array so we can mutate in-place
        // to get the pointers without any unsafe.
        let mut frequencies = [SuffixIdx(0); Self::N_TRIGRAMS];

        for i in 0..content.len() {
            let suffix = &content[i..];
            let trigram_idx = match suffix {
                [a, b, c, ..] => usize::from(*a) << 16 + usize::from(*b) << 8 + usize::from(*c),
                [a, b] => usize::from(*a) << 16 + usize::from(*b) << 8,
                [a] => usize::from(*a) << 16,
                _ => unreachable!("should not ever have an empty slice"),
            };
            frequencies[trigram_idx] += SuffixIdx(1)
        }

        frequencies.iter_mut().fold(SuffixIdx(0), |acc, freq| {
            *freq += acc;
            *freq
        });
        let pointers = frequencies;
        Self(pointers)
    }

    pub fn compress(&self) -> CompressedTrigramPointers {
        CompressedTrigramPointers::new(&self.0)
    }
}

struct CompressedTrigramPointers(EliasFano);

impl CompressedTrigramPointers {
    pub fn new(pointers: &[SuffixIdx; TrigramPointers::N_TRIGRAMS]) -> Self {
        let universe = u32::from(pointers[TrigramPointers::N_TRIGRAMS - 1]) as usize + 1;
        let mut builder = EliasFanoBuilder::new(universe, pointers.len()).unwrap();
        for &idx in pointers {
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

    // Returns an inclusive lower bound on the suffixes with the prefix needle
    fn lower_bound<T>(&self, needle: T) -> SuffixIdx
    where
        T: AsRef<[u8]>,
    {
        let idx = match needle.as_ref() {
            [a, b, c, ..] => usize::from(*a) << 16 + usize::from(*b) << 8 + usize::from(*c),
            [a, b] => usize::from(*a) << 16 + usize::from(*b) << 8,
            [a] => usize::from(*a) << 16,
            [] => 0,
        };
        SuffixIdx(self.0.select(idx) as u32)
    }

    // Returns an exclusive upper bound on the suffixes with the prefix needle
    fn upper_bound<T>(&self, needle: T) -> SuffixIdx
    where
        T: AsRef<[u8]>,
    {
        // TODO audit these saturating adds carefully at the boundary conditions
        let idx = match needle.as_ref() {
            [a, b, c, ..] => {
                (usize::from(*a) << 16 + usize::from(*b) << 8 + usize::from(*c)).saturating_add(1)
            }
            [a, b] => (usize::from(*a) << 8 + usize::from(*b)).saturating_add(1) << 8,
            [a] => usize::from(*a).saturating_add(1) << 16,
            [] => self.0.len(),
        };
        SuffixIdx(self.0.select(idx) as u32)
    }
}
