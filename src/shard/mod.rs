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
use bytes::{Buf, BufMut};
use content::ContentStore;
use derive_more::{Add, From, Into, Sub};
use docs::DocStore;
use std::io::{Read, Write};
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
            header.content.offset,
            header.content.len as u32,
        );
        let docs = DocStore::new(
            Arc::clone(&file),
            content.clone(),
            header.doc_ends.offset,
            header.doc_ends.len as u32,
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

#[derive(Clone, Debug, Default)]
pub struct ShardHeader {
    pub version: u32,
    pub flags: u32,
    pub content: SimpleSection,
    pub doc_ends: SimpleSection,
    pub sa: SimpleSection,
}

impl ShardHeader {
    const VERSION: u32 = 1;
    const HEADER_SIZE: usize = 1 << 13; /* 8192 */
    const FLAG_COMPLETE: u32 = 1 << 0;

    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ShardHeader::HEADER_SIZE).writer();
        self.write_to(&mut buf).unwrap();
        buf.into_inner()
    }
}

impl ReadWriteStream for ShardHeader {
    fn read_from<R: Read>(r: &mut R) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let version = u32::read_from(r)?;
        let flags = u32::read_from(r)?;
        let content = SimpleSection::read_from(r)?;
        let doc_ends = SimpleSection::read_from(r)?;
        let sa = SimpleSection::read_from(r)?;

        Ok(Self {
            version,
            flags,
            content,
            doc_ends,
            sa,
        })
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error> {
        let mut n = 0;
        n += self.version.write_to(w)?;
        n += self.flags.write_to(w)?;
        n += self.content.write_to(w)?;
        n += self.doc_ends.write_to(w)?;
        n += self.sa.write_to(w)?;
        Ok(n)
    }
}

// SimpleSection describes a simple range of bytes.
#[derive(Clone, Debug, Default)]
pub struct SimpleSection {
    pub offset: u64,
    pub len: u64,
}

impl ReadWriteStream for SimpleSection {
    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error> {
        let mut n = 0;
        n += self.offset.write_to(w)?;
        n += self.len.write_to(w)?;
        Ok(n)
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self, Error> {
        let offset = u64::read_from(r)?;
        let len = u64::read_from(r)?;
        Ok(Self { offset, len })
    }
}

// CompoundSection describes a range of bytes that contains variable-width items.
#[derive(Clone, Debug, Default)]
pub struct CompoundSection {
    pub data: SimpleSection,
    pub offsets: SimpleSection,
}

impl CompoundSection {
    fn offset(&self) -> u64 {
        self.data.offset
    }

    fn len(&self) -> u64 {
        self.data.len + self.offsets.len
    }

    fn end(&self) -> u64 {
        self.offset() + self.len()
    }
}

impl ReadWriteStream for CompoundSection {
    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error> {
        let mut n = self.data.write_to(w)?;
        n += self.offsets.write_to(w)?;
        Ok(n)
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self, Error> {
        let data = SimpleSection::read_from(r)?;
        let offsets = SimpleSection::read_from(r)?;
        Ok(Self { data, offsets })
    }
}

pub trait ReadWriteStream {
    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error>;
    fn read_from<R: Read>(r: &mut R) -> Result<Self, Error>
    where
        Self: Sized;
}

impl ReadWriteStream for u64 {
    fn read_from<R: Read>(r: &mut R) -> Result<Self, Error> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error> {
        w.write_all(&self.to_le_bytes())?;
        Ok(8)
    }
}

impl ReadWriteStream for u32 {
    fn read_from<R: Read>(r: &mut R) -> Result<Self, Error> {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error> {
        w.write_all(&self.to_le_bytes())?;
        Ok(4)
    }
}
