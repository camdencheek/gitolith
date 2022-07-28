use super::docs::DocID;
use super::suffix::{SuffixBlock, TrigramPointers};
use super::{Shard, ShardHeader};
use memmap2::{Mmap, MmapMut};
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;
use suffix;

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

        // Pad each file with a zero byte so each offset in the corpus
        // is unambiguously associated with a single document.
        self.file.write_all(&[0u8])?;

        // Track the offsets of each doc-ending zero byte in the concatenated corpus
        match self.doc_ends.as_slice() {
            [.., last] => self.doc_ends.push(last + 1 + doc_len as u32),
            [] => self.doc_ends.push(doc_len as u32),
        };
        Ok(DocID(self.doc_ends.len() as u32 - 1))
    }

    pub fn build(mut self) -> Result<Shard, Box<dyn std::error::Error>> {
        let (content_ptr, content_len) = Self::build_content(&self.doc_ends);
        let (doc_starts_ptr, doc_starts_len) = Self::build_docs(&mut self.file, &self.doc_ends)?;
        let (suffix_ptr, suffix_len, trigram_ptr, trigram_len) =
            Self::build_suffix_array(&mut self.file, content_ptr, content_len)?;
        Self::build_header(
            &self.file,
            content_ptr,
            content_len.into(),
            doc_starts_ptr,
            doc_starts_len.into(),
            suffix_ptr,
            suffix_len.into(),
            trigram_ptr,
            trigram_len,
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
    ) -> Result<(u64, u32, u64, u64), Box<dyn std::error::Error>> {
        let (pointers_start, pointers_len) = {
            let mmap = unsafe { Mmap::map(&*file)? };
            let content_data =
                &mmap[content_ptr as usize..content_ptr as usize + content_len as usize];

            let pointers = TrigramPointers::from_content(content_data).compress();
            let pointers_start = file.seek(SeekFrom::Current(0))?;
            let mut buf = Vec::new(); // TODO allocate this to the right capacity
            let pointers_len = pointers.serialize_into(&mut buf)?;
            file.write_all(&buf)?;
            (pointers_start, pointers_len)
        };

        let sa_start = {
            let current_position = file.seek(io::SeekFrom::Current(0))?;
            assert!(current_position == pointers_start + pointers_len as u64);

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

            let mut stypes = suffix::SuffixTypes::new(sa.len() as u32);
            let mut bins = suffix::Bins::new();
            suffix::sais(sa, &mut stypes, &mut bins, &suffix::Utf8(content_data));
            sa_start
        };
        Ok((sa_start, content_len, pointers_start, pointers_len as u64))
    }

    fn build_header(
        file: &File,
        content_ptr: u64,
        content_len: u64,
        doc_ends_ptr: u64,
        doc_ends_len: u64,
        sa_ptr: u64,
        sa_len: u64,
        trigrams_ptr: u64,
        trigrams_len: u64,
    ) -> Result<ShardHeader, io::Error> {
        let header = ShardHeader {
            version: ShardHeader::VERSION,
            flags: ShardHeader::FLAG_COMPLETE,
            _padding: 0,
            content_ptr,
            content_len,
            doc_ends_ptr,
            doc_ends_len,
            sa_ptr,
            sa_len,
            trigrams_ptr,
            trigrams_len,
        };
        file.write_all_at(&header.to_bytes(), 0)?;
        Ok(header)
    }
}

// TODO use u64::next_multiple_of() when #88581 stabilizes
fn next_multiple_of(lhs: u64, rhs: u64) -> u64 {
    match lhs % rhs {
        0 => lhs,
        r => lhs + (rhs - r),
    }
}
