pub mod builder;
pub mod cached;
pub mod content;
pub mod suffix;

use anyhow::Error;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
pub mod docs;
// use super::cache::Cache;
use super::shard::suffix::SuffixArrayStore;
use content::ContentStore;
use derive_more::{Add, From, Into, Sub};
use docs::DocStore;
use std::io::Write;
use std::sync::Arc;

#[derive(Copy, Clone, From, Into, Add, Sub, PartialEq, Eq, Hash, Debug)]
pub struct ShardID(pub u16);

#[derive(Clone)]
pub struct Shard {
    pub header: ShardHeader,
    pub docs: DocStore,
    pub suffixes: SuffixArrayStore,
}

impl Shard {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let f = File::open(path)?;
        Self::from_file(f)
    }

    fn from_file(file: File) -> Result<Self, Error> {
        let mut buf = [0u8; ShardHeader::HEADER_SIZE];
        file.read_at(&mut buf[..], 0)?;
        let header = ShardHeader::from_bytes(&buf[..])?;

        let file = Arc::new(file);
        let content = ContentStore::new(
            Arc::clone(&file),
            header.content_ptr,
            header.content_len as u32,
        );
        let docs = DocStore::new(
            Arc::clone(&file),
            content.clone(),
            header.doc_ends_ptr,
            header.doc_ends_len as u32,
        );
        let suffixes = SuffixArrayStore::new(
            Arc::clone(&file),
            content,
            header.sa_ptr,
            header.sa_len as u32,
            header.trigrams_ptr,
            header.trigrams_len,
        );

        Ok(Self {
            header,
            docs,
            suffixes,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ShardHeader {
    pub version: u16,
    pub flags: u16,
    pub _padding: u32,
    pub content_ptr: u64,
    pub content_len: u64,
    pub doc_ends_ptr: u64,
    pub doc_ends_len: u64,
    pub sa_ptr: u64,
    pub sa_len: u64,
    pub trigrams_ptr: u64,
    pub trigrams_len: u64,
}

impl ShardHeader {
    const VERSION: u16 = 1;
    const HEADER_SIZE: usize = 1 << 13; /* 8192 */
    const FLAG_COMPLETE: u16 = 1 << 0;

    const OFFSETS_LEN: usize = 256;

    // TODO add a first character index
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::HEADER_SIZE as usize);
        buf.write(&self.version.to_le_bytes()).unwrap();
        buf.write(&self.flags.to_le_bytes()).unwrap();
        buf.write(&self._padding.to_le_bytes()).unwrap();
        buf.write(&self.content_ptr.to_le_bytes()).unwrap();
        dbg!(&self.content_ptr);
        buf.write(&self.content_len.to_le_bytes()).unwrap();
        buf.write(&self.doc_ends_ptr.to_le_bytes()).unwrap();
        buf.write(&self.doc_ends_len.to_le_bytes()).unwrap();
        buf.write(&self.sa_ptr.to_le_bytes()).unwrap();
        buf.write(&self.sa_len.to_le_bytes()).unwrap();
        buf.write(&self.trigrams_ptr.to_le_bytes()).unwrap();
        buf.write(&self.trigrams_len.to_le_bytes()).unwrap();
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, Error> {
        Ok(Self {
            version: u16::from_le_bytes(buf[0..2].try_into()?),
            flags: u16::from_le_bytes(buf[2..4].try_into()?),
            _padding: u32::from_le_bytes(buf[4..8].try_into()?),
            content_ptr: u64::from_le_bytes(buf[8..16].try_into()?),
            content_len: u64::from_le_bytes(buf[16..24].try_into()?),
            doc_ends_ptr: u64::from_le_bytes(buf[24..32].try_into()?),
            doc_ends_len: u64::from_le_bytes(buf[32..40].try_into()?),
            sa_ptr: u64::from_le_bytes(buf[40..48].try_into()?),
            sa_len: u64::from_le_bytes(buf[48..56].try_into()?),
            trigrams_ptr: u64::from_le_bytes(buf[56..64].try_into()?),
            trigrams_len: u64::from_le_bytes(buf[64..72].try_into()?),
        })
    }
}

impl Default for ShardHeader {
    fn default() -> Self {
        ShardHeader {
            version: Self::VERSION,
            flags: 0,
            _padding: 0,
            content_ptr: 0,
            content_len: 0,
            doc_ends_ptr: 0,
            doc_ends_len: 0,
            sa_ptr: 0,
            sa_len: 0,
            trigrams_ptr: 0,
            trigrams_len: 0,
        }
    }
}
