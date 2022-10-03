pub mod builder;
pub mod cached;
pub mod content;
pub mod docs;
pub mod file;
pub mod suffix;

use bytes::Buf;
use content::ContentStore;
use file::ShardHeader;

use self::file::ReadWriteStream;
use anyhow::Error;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

// use super::cache::Cache;
use super::shard::suffix::SuffixArrayStore;
use derive_more::{Add, From, Into, Sub};
use docs::DocStore;
use std::sync::Arc;

#[derive(Copy, Clone, From, Into, Add, Sub, PartialEq, Eq, Hash, Debug)]
pub struct ShardID(pub u16);

impl From<ShardID> for u64 {
    fn from(shard_id: ShardID) -> Self {
        shard_id.0 as u64
    }
}

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
        let header = ShardHeader::read_from(&mut buf[..].reader())?;

        let file = Arc::new(file);
        let content = ContentStore::new(
            Arc::clone(&file),
            header.docs.data.offset,
            header.docs.data.len as u32,
        );
        let docs = DocStore::new(
            Arc::clone(&file),
            content.clone(),
            header.docs.offsets.offset,
            header.docs.offsets.len as u32,
        );
        let suffixes =
            SuffixArrayStore::new(Arc::clone(&file), header.sa.offset, header.sa.len as u32);

        Ok(Self {
            header,
            docs,
            suffixes,
        })
    }
}
