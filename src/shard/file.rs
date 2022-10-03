use std::{
    fs::File,
    io::{Read, Write},
};

use anyhow::Error;
use bytes::BufMut;

struct ShardFile {
    file: File,
    header: ShardHeader,
}

#[derive(Clone, Debug, Default)]
pub struct ShardHeader {
    pub version: u32,
    pub flags: u32,
    pub docs: CompoundSection,
    pub sa: SimpleSection,
}

impl ShardHeader {
    pub const VERSION: u32 = 1;
    pub const HEADER_SIZE: usize = 1 << 13; /* 8192 */
    pub const FLAG_COMPLETE: u32 = 1 << 0;

    pub fn to_bytes(&self) -> Vec<u8> {
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
        let docs = CompoundSection::read_from(r)?;
        let sa = SimpleSection::read_from(r)?;

        Ok(Self {
            version,
            flags,
            docs,
            sa,
        })
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error> {
        let mut n = 0;
        n += self.version.write_to(w)?;
        n += self.flags.write_to(w)?;
        n += self.docs.write_to(w)?;
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
