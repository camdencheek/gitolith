use super::content::{ContentIdx, ContentStore};
use super::docs::DocStore;
use super::{Shard, ShardHeader};
use memmap2::{Mmap, MmapMut};
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::rc::Rc;
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

    pub fn build(mut self) -> Result<Shard, io::Error> {
        self.write_doc_ends()?;
        self.write_suffix_array()?;
        let header = self.write_header()?;

        let file = Rc::new(self.file);
        let content = ContentStore::new(
            Rc::clone(&file),
            ShardHeader::HEADER_SIZE as u64,
            *self.doc_ends.last().unwrap_or(&0) as u32 + 1,
        );

        let doc_ends_ptr = header.doc_ends_ptr;
        let doc_ends_len = header.doc_ends_ptr as usize;
        let docs = DocStore::new(doc_ends_ptr, doc_ends_len, Rc::clone(&file), content);

        Ok(Shard { header, docs })
    }

    fn write_doc_ends(&mut self) -> Result<(), io::Error> {
        // Write all the doc ends to the buffer
        let mut buf = BufWriter::new(&self.file);
        for doc_end in self.doc_ends.iter() {
            buf.write(&doc_end.to_le_bytes())?;
        }
        buf.flush()?;
        Ok(())
    }

    fn write_suffix_array(&mut self) -> Result<(), io::Error> {
        let data_mmap = unsafe { Mmap::map(&self.file)? };

        // TODO all these offsets should be defined in one place
        let content_start = ShardHeader::HEADER_SIZE;
        let content_end = content_start + *self.doc_ends.last().unwrap_or(&0) as usize;
        let content = &data_mmap[content_start..content_end];

        let index_mmap = MmapMut::map_anon(content.len() * std::mem::size_of::<u32>())?;
        let sa = unsafe {
            std::slice::from_raw_parts_mut(index_mmap[..].as_ptr() as *mut u32, content.len())
        };

        let mut stypes = suffix::SuffixTypes::new(sa.len() as u32);
        let mut bins = suffix::Bins::new();
        suffix::sais(sa, &mut stypes, &mut bins, &suffix::Utf8(content));

        let mut buf = BufWriter::new(&self.file);
        for suffix in sa.into_iter() {
            buf.write(&suffix.to_le_bytes())?;
        }

        let find_char_offset =
            |char: u8| -> u32 { sa.partition_point(|idx| content[*idx as usize] < char) as u32 };

        for c in u8::MIN..=u8::MAX {
            buf.write(&find_char_offset(c).to_le_bytes())?;
        }
        buf.flush()?;
        Ok(())
    }

    fn write_header(&self) -> Result<ShardHeader, io::Error> {
        let content_ptr = ShardHeader::HEADER_SIZE as u64;
        let content_len = *self.doc_ends.last().unwrap_or(&0) as u64 + 1;
        let doc_ends_ptr = content_ptr + content_len;
        let doc_ends_len = self.doc_ends.len() as u64;
        let sa_ptr = doc_ends_ptr + doc_ends_len * std::mem::size_of::<u32>() as u64;
        let sa_len = content_len;
        let offsets_ptr = sa_ptr + sa_len * std::mem::size_of::<u32>() as u64;

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
            offsets_ptr,
        };

        self.file.write_at(&header.to_bytes(), 0)?;
        Ok(header)
    }
}
