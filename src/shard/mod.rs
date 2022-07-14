use memmap::MmapMut;
use std::fs::File;
use std::io::{Error, Seek, SeekFrom, Write};
use std::iter::Iterator;
use std::path::Path;
use suffix;

pub type ShardID = u32;

pub struct Shard {}

impl Shard {
    pub fn new<'a, T: Iterator<Item = &'a Vec<u8>>>(
        id: ShardID,
        path: &Path,
        docs: T,
    ) -> Result<Self, Error> {
        let mut f = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        println!("opened shard");

        // Create a new header and write it to reserve the space
        let mut header = ShardHeader {
            id,
            ..ShardHeader::default()
        };
        f.seek(SeekFrom::Start(ShardHeader::HEADER_SIZE as u64));

        // Content starts immediately after the header
        header.content_ptr = ShardHeader::HEADER_SIZE as u64;

        // Create a peekable iterator so we can check whether the next
        // value would push us over the 4GB limit.
        let mut docs = docs.peekable();

        let mut current_doc_start: u64 = 0;
        let mut doc_starts: Vec<u64> = Vec::with_capacity(docs.size_hint().0);

        let zero_byte: [u8; 1] = [0; 1];
        while let Some(doc) =
            docs.next_if(|peeked| header.content_len + peeked.len() as u64 + 1 <= u32::MAX.into())
        {
            doc_starts.push(current_doc_start);
            current_doc_start += doc.len() as u64;

            f.write_all(&doc)?;
            f.write_all(&zero_byte)?; // zero byte at end of each document

            header.content_len += doc.len() as u64 + 1;
        }

        let mut buf: Vec<u8> = Vec::with_capacity(doc_starts.len() * 8);
        for doc_start in doc_starts {
            buf.extend_from_slice(&doc_start.to_le_bytes());
        }
        f.write_all(&buf)?;
        header.doc_starts_ptr = header.content_ptr + header.content_len;
        header.doc_starts_len = buf.len() as u64;

        header.sa_ptr = header.doc_starts_ptr + header.doc_starts_len;
        header.sa_len = header.content_len * std::mem::size_of::<u32>() as u64;

        println!("setting length to {}", header.sa_ptr + header.sa_len);
        f.set_len(header.sa_ptr + header.sa_len);

        println!("initialized shard");

        let mmap = unsafe { MmapMut::map_mut(&f)? };
        let content =
            &mmap[header.content_ptr as usize..(header.content_ptr + header.content_len) as usize];
        let sa = unsafe {
            std::slice::from_raw_parts_mut(
                mmap[header.sa_ptr as usize..].as_ptr() as *mut u32,
                header.content_len as usize,
            )
        };

        let mut stypes = suffix::SuffixTypes::new(sa.len() as u32);
        let mut bins = suffix::Bins::new();
        suffix::sais(sa, &mut stypes, &mut bins, &suffix::Utf8(content));
        Ok(Shard {})
    }
}

pub struct Document<'a> {
    content: &'a [u8],
}

#[repr(C)]
pub struct ShardHeader {
    pub version: u16,
    pub flags: u16,
    pub id: ShardID,
    pub content_ptr: u64,
    pub content_len: u64,
    pub doc_starts_ptr: u64,
    pub doc_starts_len: u64,
    pub sa_ptr: u64,
    pub sa_len: u64,
}

impl ShardHeader {
    const VERSION: u16 = 1;
    const HEADER_SIZE: usize = 1 << 13; /* 8192 */
    const FLAG_INCOMPLETE: u16 = 1 << 0;

    pub fn to_bytes(&self) -> &[u8; Self::HEADER_SIZE] {
        todo!();
    }

    pub fn from_bytes() -> Result<Self, ()> {
        todo!();
    }
}

impl Default for ShardHeader {
    fn default() -> Self {
        ShardHeader {
            version: Self::VERSION,
            flags: Self::FLAG_INCOMPLETE,
            id: 0,
            content_ptr: 0,
            content_len: 0,
            doc_starts_ptr: 0,
            doc_starts_len: 0,
            sa_ptr: 0,
            sa_len: 0,
        }
    }
}
