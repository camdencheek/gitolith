use super::docs::DocID;
use super::file::{CompoundSection, ShardHeader, SimpleSection};
use super::suffix::SuffixBlock;
use super::Shard;
use crate::strcmp::AsciiLowerIter;
use anyhow::Error;
use memmap2::MmapMut;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;
use suffix::table;

#[derive(Debug)]
pub struct ShardBuilder {
    file: File,
    // the locations of the zero-bytes following each document,
    // relative to the start of the content.
    doc_ends: Vec<u32>,
}

impl ShardBuilder {
    pub fn new(path: &Path) -> Result<Self, io::Error> {
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Reserve space for the header before starting document content
        file.seek(SeekFrom::Start(ShardHeader::HEADER_SIZE as u64))?;

        Ok(Self {
            file,
            doc_ends: Vec::new(),
        })
    }

    // Adds a doc to the index and returns its ID
    pub fn add_doc<T>(&mut self, mut doc: T) -> Result<DocID, io::Error>
    where
        T: Read,
    {
        // Copy the document into the index
        let doc_len = io::copy(&mut doc, &mut self.file)?;

        // Track the offsets of the exclusive end offset of each document in the corpus
        match self.doc_ends.as_slice() {
            [.., last] => self.doc_ends.push(last + doc_len as u32),
            [] => self.doc_ends.push(doc_len as u32),
        };
        Ok(DocID(self.doc_ends.len() as u32 - 1))
    }

    pub fn build(mut self) -> Result<Shard, Error> {
        let (content_ptr, content_len) = Self::build_content(&self.doc_ends);
        let (doc_starts_ptr, doc_starts_len) = Self::build_docs(&mut self.file, &self.doc_ends)?;
        let (suffix_ptr, suffix_len) =
            Self::build_suffix_array(&mut self.file, content_ptr, content_len)?;
        Self::build_header(
            &self.file,
            content_ptr,
            content_len.into(),
            doc_starts_ptr,
            doc_starts_len.into(),
            suffix_ptr,
            suffix_len.into(),
        )?;
        Ok(Shard::from_file(self.file)?)
    }

    fn build_content(doc_ends: &Vec<u32>) -> (u64, u32) {
        // The data on disk is build incrementally during add_doc,
        // so just return the content range here
        (
            ShardHeader::HEADER_SIZE as u64,
            *doc_ends.last().unwrap_or(&0),
        )
    }

    fn build_docs(file: &mut File, doc_ends: &Vec<u32>) -> Result<(u64, u32), io::Error> {
        // Write all the doc ends to the buffer
        let start = file.seek(io::SeekFrom::Current(0))?;
        let mut buf = BufWriter::new(file);
        for doc_end in doc_ends.iter() {
            buf.write_all(&doc_end.to_le_bytes())?;
        }
        buf.flush()?;
        Ok((start, doc_ends.len() as u32))
    }

    fn build_suffix_array(
        file: &mut File,
        content_ptr: u64,
        content_len: u32,
    ) -> Result<(u64, u32), Error> {
        let sa_start = {
            let current_position = file.seek(io::SeekFrom::Current(0))?;

            // Round up to the nearest block size so we have aligned blocks for our suffix array
            let sa_start = next_multiple_of(current_position, SuffixBlock::SIZE_BYTES as u64);
            let sa_end = sa_start + content_len as u64 * std::mem::size_of::<u32>() as u64;

            // Round file length to the next block size and move the cursor to the end of the file
            file.set_len(next_multiple_of(sa_end, SuffixBlock::SIZE_BYTES as u64))?;
            file.seek(io::SeekFrom::End(0))?;

            // Reopen mmap after extending file
            let mmap = unsafe { MmapMut::map_mut(&*file)? };
            let content_data =
                &mmap[content_ptr as usize..content_ptr as usize + content_len as usize];

            let sa = unsafe {
                std::slice::from_raw_parts_mut(
                    mmap[sa_start as usize..].as_ptr() as *mut u32,
                    content_len as usize,
                )
            };

            let mut stypes = table::SuffixTypes::new(sa.len() as u32);
            let mut bins = table::Bins::new();
            table::sais(sa, &mut stypes, &mut bins, &CIBytes(content_data));
            sa_start
        };
        Ok((sa_start, content_len))
    }

    fn build_header(
        file: &File,
        content_ptr: u64,
        content_len: u64,
        doc_ends_ptr: u64,
        doc_ends_len: u64,
        sa_ptr: u64,
        sa_len: u64,
    ) -> Result<ShardHeader, io::Error> {
        let header = ShardHeader {
            version: ShardHeader::VERSION,
            flags: ShardHeader::FLAG_COMPLETE,
            docs: CompoundSection {
                data: SimpleSection {
                    offset: content_ptr,
                    len: content_len,
                },
                offsets: SimpleSection {
                    offset: doc_ends_ptr,
                    len: doc_ends_len,
                },
            },
            sa: SimpleSection {
                offset: sa_ptr,
                len: sa_len,
            },
        };
        file.write_all_at(&header.to_bytes(), 0)?;
        Ok(header)
    }
}

// NOTE: we can use u64::next_multiple_of() when #88581 stabilizes
fn next_multiple_of(lhs: u64, rhs: u64) -> u64 {
    match lhs % rhs {
        0 => lhs,
        r => lhs + (rhs - r),
    }
}

pub struct CIBytes<'s>(pub &'s [u8]);

impl<'s> suffix::table::Text for CIBytes<'s> {
    type IdxChars =
        std::iter::Enumerate<AsciiLowerIter<std::iter::Copied<std::slice::Iter<'s, u8>>>>;

    #[inline]
    fn len(&self) -> u32 {
        self.0.len() as u32
    }

    #[inline]
    fn prev(&self, i: u32) -> (u32, u32) {
        (i - 1, self.0[i as usize - 1].to_ascii_lowercase() as u32)
    }

    #[inline]
    fn char_at(&self, i: u32) -> u32 {
        self.0[i as usize].to_ascii_lowercase() as u32
    }

    fn char_indices(
        &self,
    ) -> std::iter::Enumerate<AsciiLowerIter<std::iter::Copied<std::slice::Iter<'s, u8>>>> {
        AsciiLowerIter::new(self.0.iter().copied()).enumerate()
    }

    fn wstring_equal(&self, stypes: &table::SuffixTypes, w1: u32, w2: u32) -> bool {
        let w1chars = self.0[w1 as usize..]
            .iter()
            .map(u8::to_ascii_lowercase)
            .enumerate();
        let w2chars = self.0[w2 as usize..]
            .iter()
            .map(u8::to_ascii_lowercase)
            .enumerate();
        for ((i1, c1), (i2, c2)) in w1chars.zip(w2chars) {
            let (i1, i2) = (w1 + i1 as u32, w2 + i2 as u32);
            if c1 != c2 || !stypes.equal(i1, i2) {
                return false;
            }
            if i1 > w1 && (stypes.is_valley(i1) || stypes.is_valley(i2)) {
                return true;
            }
        }
        // At this point, we've exhausted either `w1` or `w2`, which means the
        // next character for one of them should be the sentinel. Since
        // `w1 != w2`, only one string can be exhausted. The sentinel is never
        // equal to another character, so we can conclude that the wstrings
        // are not equal.
        false
    }
}
