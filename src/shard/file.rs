use std::{
    fs::File,
    io::{Cursor, Read, Write},
    os::unix::prelude::FileExt,
    sync::Arc,
};

use anyhow::Error;

use super::{
    docs::{ContentIdx, DocEnds, DocID},
    suffix::{SuffixBlock, SuffixBlockID},
    Trigram,
};

pub type ShardStore = Arc<dyn ShardBackend + Send + Sync>;

pub trait ShardBackend {
    fn header(&self) -> &ShardHeader;
    fn get_doc_ends(&self) -> Result<Arc<DocEnds>, Error>;
    fn get_doc(&self, doc_id: DocID, doc_ends: &DocEnds) -> Result<Arc<[u8]>, Error>;
    fn get_suffix_block(&self, block_id: SuffixBlockID) -> Result<Arc<SuffixBlock>, Error>;
    fn get_trigrams(&self) -> Result<Arc<[(Trigram, u32)]>, Error>;
}

pub struct ShardFile {
    pub file: File,
    pub header: ShardHeader,
}

impl ShardFile {
    pub fn from_file(file: File) -> Result<Self, Error> {
        let mut buf = [0u8; ShardHeader::HEADER_SIZE];
        file.read_at(&mut buf[..], 0)?;
        let header = ShardHeader::read_from(&mut Cursor::new(buf))?;
        Ok(Self { file, header })
    }
}

impl ShardBackend for ShardFile {
    // TODO these methods are currently written to avoid any unsafe or nightly-only features. There
    // are a few extra copies in here because of that, but we should consider things like
    // Arc::new_zeroed_slice, Arc::get_mut_unchecked, etc. Some quick tests showed that
    // perf difference were mostly trivial compared to disk accesses, but testing was not thorough.

    fn header(&self) -> &ShardHeader {
        &self.header
    }

    fn get_doc_ends(&self) -> Result<Arc<DocEnds>, Error> {
        let mut buf = vec![0u8; self.header.content.offsets.len as usize];
        self.file
            .read_exact_at(&mut buf, self.header.content.offsets.offset)?;

        let chunks = buf.chunks_exact(std::mem::size_of::<u32>());
        assert!(chunks.remainder().is_empty());

        Ok(Arc::new(DocEnds::new(
            chunks
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .map(ContentIdx::from)
                .collect(),
        )))
    }

    fn get_doc(&self, doc_id: DocID, doc_ends: &DocEnds) -> Result<Arc<[u8]>, Error> {
        let range = doc_ends.content_range(doc_id);
        let doc_start = self.header.content.data.offset + u64::from(range.start);
        let doc_len = usize::from(range.end) - usize::from(range.start);
        let mut buf = vec![0u8; doc_len];
        self.file.read_exact_at(&mut buf, doc_start)?;
        Ok(buf.into())
    }

    fn get_suffix_block(&self, block_id: SuffixBlockID) -> Result<Arc<SuffixBlock>, Error> {
        let block_start =
            self.header.sa.offset + u64::from(block_id) * SuffixBlock::SIZE_BYTES as u64;
        let mut buf = [0u8; SuffixBlock::SIZE_BYTES];
        self.file.read_exact_at(&mut buf, block_start)?;

        let chunks = buf.chunks_exact(std::mem::size_of::<u32>());
        assert!(chunks.remainder().is_empty());

        let mut block = Arc::new(SuffixBlock::default());
        let block_ref = Arc::get_mut(&mut block).unwrap();
        for (i, chunk) in chunks.enumerate() {
            block_ref.0[i] = ContentIdx(u32::from_le_bytes(chunk.try_into()?));
        }
        Ok(block)
    }

    fn get_trigrams(&self) -> Result<Arc<[(Trigram, u32)]>, Error> {
        assert!(self.header().trigrams.len % 7 == 0);

        let mut buf = vec![0u8; self.header().trigrams.len as usize];
        self.file
            .read_exact_at(&mut buf, self.header().trigrams.offset)?;

        Ok(buf
            .chunks_exact(7)
            .map(|chunk| {
                let mut trigram = [0u8; 3];
                trigram.copy_from_slice(&chunk[..3]);
                let mut count_bytes = [0u8; 4];
                count_bytes.copy_from_slice(&chunk[3..]);
                (trigram, u32::from_le_bytes(count_bytes))
            })
            .collect::<Vec<_>>()
            .into())
    }
}

#[derive(Clone, Debug, Default)]
pub struct ShardHeader {
    pub version: u32,
    pub flags: u32,
    pub content: CompoundSection,
    pub trigrams: SimpleSection,
    pub sa: SimpleSection,
}

impl ShardHeader {
    pub const VERSION: u32 = 1;
    pub const HEADER_SIZE: usize = 1 << 13; /* 8192 */
    pub const FLAG_COMPLETE: u32 = 1 << 0;

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::with_capacity(ShardHeader::HEADER_SIZE));
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
        let content = CompoundSection::read_from(r)?;
        let trigrams = SimpleSection::read_from(r)?;
        let sa = SimpleSection::read_from(r)?;

        Ok(Self {
            version,
            flags,
            content,
            trigrams,
            sa,
        })
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<usize, Error> {
        let mut n = 0;
        n += self.version.write_to(w)?;
        n += self.flags.write_to(w)?;
        n += self.content.write_to(w)?;
        n += self.trigrams.write_to(w)?;
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
