use super::suffix::{SuffixBlock, TrigramPointers};
use super::{Shard, ShardHeader};
use memmap2::{Mmap, MmapMut};
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;
use suffix;

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
    pub fn add_doc<T>(&mut self, mut doc: T) -> Result<u32, io::Error>
    where
        T: Read,
    {
        // Copy the document into the index
        let doc_len = io::copy(&mut doc, &mut self.file)?;

        // Pad each file with a zero byte so each offset in the corpus
        // is unambiguously associated with a single document.
        const ZERO_BYTE: [u8; 1] = [0; 1];
        self.file.write_all(&ZERO_BYTE[..])?;

        // Track the offsets of each document in the concatenated corpus
        match self.doc_ends.as_slice() {
            [.., last] => self.doc_ends.push(last + doc_len as u32 + 1),
            [] => self.doc_ends.push(doc_len as u32 + 1),
        };
        Ok(self.doc_ends.len() as u32 - 1)
    }

    pub fn build(mut self) -> Result<Shard, Box<dyn std::error::Error>> {
        let header_range = 0..ShardHeader::HEADER_SIZE as u64;
        let content_range =
            header_range.end..header_range.end + *self.doc_ends.last().unwrap_or(&0) as u64;
        let doc_ends_range = self.write_doc_ends()?;
        let suffix_array_range = self.write_suffix_array(content_range.clone())?;
        let trigram_pointers_range = self.write_trigram_pointers(content_range.clone())?;

        let header = self.write_header(
            content_range,
            doc_ends_range,
            suffix_array_range,
            trigram_pointers_range,
        )?;

        Ok(Shard::from_file(self.file)?)
    }

    // Writes the collected doc ends to the file, returning the byte range in the file that
    // contains the doc ends.
    fn write_doc_ends(&mut self) -> Result<Range<u64>, io::Error> {
        // Write all the doc ends to the buffer
        let start = self.file.seek(io::SeekFrom::Current(0))?;
        let mut buf = BufWriter::new(&self.file);
        for doc_end in self.doc_ends.iter() {
            buf.write_all(&doc_end.to_le_bytes())?;
        }
        buf.flush()?;
        Ok(start..(self.doc_ends.len() * std::mem::size_of::<u32>()) as u64)
    }

    fn write_suffix_array(&mut self, content_range: Range<u64>) -> Result<Range<u64>, io::Error> {
        let current_position = self.file.seek(io::SeekFrom::Current(0))?;

        // Round up to the nearest block size so we have aligned blocks for our suffix array
        let sa_start = current_position
            + (SuffixBlock::SIZE_BYTES as u64 - current_position % SuffixBlock::SIZE_BYTES as u64);
        let sa_end = sa_start
            + (content_range.end - content_range.start) * std::mem::size_of::<u32>() as u64;

        self.file.set_len(sa_end)?;

        let mmap = unsafe { MmapMut::map_mut(&self.file)? };
        let content = &mmap[content_range.start as usize..content_range.end as usize];
        let sa = unsafe {
            std::slice::from_raw_parts_mut(
                mmap[sa_start as usize..].as_ptr() as *mut u32,
                content.len(),
            )
        };

        let mut stypes = suffix::SuffixTypes::new(sa.len() as u32);
        let mut bins = suffix::Bins::new();
        suffix::sais(sa, &mut stypes, &mut bins, &suffix::Utf8(content));
        Ok(sa_start..sa_end)
    }

    fn write_trigram_pointers(
        &mut self,
        content_range: Range<u64>,
    ) -> Result<Range<u64>, Box<dyn std::error::Error>> {
        let mmap = unsafe { Mmap::map(&self.file)? };
        let content = &mmap[content_range.start as usize..content_range.end as usize];
        let pointers = TrigramPointers::from_content(content);
        let compressed_pointers = pointers.compress();
        let pointers_start = self.file.seek(SeekFrom::Current(0))?;
        let pointers_len = compressed_pointers.serialize_into(&self.file)?;
        Ok(pointers_start..pointers_start + pointers_len as u64)
    }

    fn write_header(
        &self,
        content_range: Range<u64>,
        doc_ends_range: Range<u64>,
        suffixes_range: Range<u64>,
        trigram_pointers_range: Range<u64>,
    ) -> Result<ShardHeader, io::Error> {
        let header = ShardHeader {
            version: ShardHeader::VERSION,
            flags: ShardHeader::FLAG_COMPLETE,
            _padding: 0,
            content_ptr: content_range.start,
            content_len: content_range.end - content_range.start,
            doc_ends_ptr: doc_ends_range.start,
            doc_ends_len: (doc_ends_range.end - doc_ends_range.start)
                / std::mem::size_of::<u32>() as u64,
            sa_ptr: suffixes_range.start,
            sa_len: (suffixes_range.end - suffixes_range.start) / std::mem::size_of::<u32>() as u64,
            trigram_pointers_ptr: trigram_pointers_range.start,
            trigram_pointers_len: trigram_pointers_range.end - trigram_pointers_range.start,
        };

        self.file.write_at(&header.to_bytes(), 0)?;
        Ok(header)
    }
}
