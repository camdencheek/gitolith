mod regex;
mod suffix;

pub use self::regex::*;
use self::suffix::SuffixIdx;
use ::regex::bytes::Regex;
use ::suffix as suf;
use crossbeam;
use itertools::{Itertools, Product};
use memmap::{Mmap, MmapMut};
use rayon;
use regex_syntax::hir::{self, Hir, HirKind};
use std::fs::File;
use std::io::{self, Error, Read, Seek, SeekFrom, Write};
use std::iter::Iterator;
use std::ops::{Range, RangeInclusive};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::thread;

pub type ShardID = u32;
pub type DocID = u32;
pub type SuffixID = u32;

pub struct Shard {
    pub header: ShardHeader,
    raw: Mmap,
}

impl Shard {
    pub fn new<'a, T: Iterator<Item = U>, U: Read>(
        id: ShardID,
        path: &Path,
        docs: T,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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

        let mut current_doc_start: u32 = 0;
        let mut doc_starts: Vec<u32> = Vec::with_capacity(docs.size_hint().0);

        let zero_byte: [u8; 1] = [0; 1];
        for mut doc in docs {
            let doc_len = io::copy(&mut doc, &mut f)?;
            if header.content_len + doc_len + 1 >= u32::MAX.into() {
                // truncate file back to the length before hitting the limit
                f.set_len(header.content_ptr + current_doc_start as u64);
                break;
            }

            doc_starts.push(current_doc_start);
            header.content_len += doc_len;

            f.write_all(&zero_byte)?; // zero byte at end of each document
            header.content_len += zero_byte.len() as u64;

            // set current_doc_start for the next doc
            current_doc_start = header.content_len as u32;
        }

        header.doc_starts_ptr =
            header.content_ptr + header.content_len * std::mem::size_of::<u8>() as u64;
        let mut buf: Vec<u8> = Vec::with_capacity(doc_starts.len() * 8);
        for doc_start in &doc_starts {
            buf.extend_from_slice(&doc_start.to_le_bytes());
        }
        f.write_all(&buf)?;
        header.doc_starts_len = doc_starts.len() as u64;

        header.sa_ptr = header.doc_starts_ptr + buf.len() as u64;
        header.sa_len = header.content_len;

        let file_size = header.sa_ptr + header.sa_len * std::mem::size_of::<u32>() as u64;
        dbg!(file_size);
        f.set_len(file_size);

        dbg!(&header);
        f.write_at(&header.to_bytes(), 0)?;
        f.seek(SeekFrom::Start(0))?;

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

        println!("opened mmap");

        let mut stypes = suf::SuffixTypes::new(sa.len() as u32);
        let mut bins = suf::Bins::new();
        suf::sais(sa, &mut stypes, &mut bins, &suf::Utf8(content));

        println!("built suffix array");
        header.flags = 0; // unset incomplete flag
        f.write_at(&header.to_bytes(), 0)?;

        Self::from_mmap(mmap.make_read_only()?)
    }

    pub fn open(id: ShardID, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            header: ShardHeader::from_bytes(&mmap)?,
            raw: mmap,
        })
    }

    pub fn content(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.raw[self.header.content_ptr as usize..].as_ptr(),
                self.header.content_len as usize,
            )
        }
    }

    pub fn doc_starts(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.raw[self.header.doc_starts_ptr as usize..].as_ptr() as *const u32,
                self.header.doc_starts_len as usize,
            )
        }
    }

    pub fn sa(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.raw[self.header.sa_ptr as usize..].as_ptr() as *const u32,
                self.header.sa_len as usize,
            )
        }
    }

    pub fn suffix(&self, idx: u32) -> &[u8] {
        &self.content()[idx as usize..]
    }

    pub fn doc_content_range(&self, id: u32) -> Range<usize> {
        self.doc_start(id) as usize..self.doc_end(id) as usize
    }

    pub fn doc_start(&self, id: u32) -> u32 {
        let starts = self.doc_starts();
        starts[id as usize]
    }

    pub fn doc_end(&self, id: u32) -> u32 {
        match self.doc_starts().get(id as usize + 1) {
            // We subtract one to remove the trailing zero byte
            Some(&next_start) => next_start - 1,
            None => self.header.content_len as u32 - 1,
        }
    }

    pub fn doc_content(&self, id: u32) -> &[u8] {
        &self.content()[self.doc_content_range(id)]
    }

    pub fn doc_from_suffix(&self, suffix: u32) -> u32 {
        let starts = self.doc_starts();
        let (mut low, mut high) = (0usize, starts.len());
        while low != high {
            let mid = (low + high) / 2;
            if self.doc_start(mid as u32) > suffix {
                high = mid - 1
            } else if self.doc_end(mid as u32) < suffix {
                low = mid + 1
            } else {
                return mid as u32;
            }
        }
        low as u32
    }

    // returns a slice of all prefixes that start with the literal needle
    pub fn sa_prefixes(&self, needle: &[u8]) -> &[u32] {
        &self.sa()[self.sa_find_start(needle) as usize..self.sa_find_end(needle) as usize]
    }

    pub fn sa_range<T>(&self, r: RangeInclusive<T>) -> &[u32]
    where
        T: AsRef<[u8]> + Ord,
    {
        debug_assert!(r.start() <= r.end());
        &self.sa()[self.sa_find_start(r.start().as_ref()) as usize
            ..self.sa_find_end(r.end().as_ref()) as usize]
    }

    // finds the index of the first suffix whose prefix is greater than or equal to needle
    pub fn sa_find_start(&self, needle: &[u8]) -> SuffixIdx {
        let sa = self.sa();
        let content = self.content();
        sa.partition_point(|idx| {
            let suf_start = *idx as usize;
            let suf_end = usize::min(suf_start + needle.len(), content.len());
            let suf = &content[suf_start..suf_end];
            suf < needle
        }) as SuffixIdx
    }

    // finds the index of the first suffix whose prefix is greater than needle
    pub fn sa_find_end(&self, needle: &[u8]) -> SuffixIdx {
        let sa = self.sa();
        let content = self.content();
        sa.partition_point(|&idx| {
            let suf_start = idx as usize;
            let suf_end = usize::min(suf_start + needle.len(), content.len());
            let suf = &content[suf_start..suf_end];
            suf <= needle
        }) as SuffixIdx
    }

    pub fn docs(&self) -> DocumentIterator<'_> {
        DocumentIterator::new(self)
    }

    pub fn search_skip_index(&self, re: Regex) -> Vec<DocMatches> {
        // Quick and dirty work-stealing with rayon.
        // TODO would be nice if this returned deterministic results.
        // TODO would be nice if this iterated over results rather than collected them.
        // TODO make this respect a limit.
        fn search_docs(s: &Shard, re: &Regex, range: Range<u32>) -> Vec<DocMatches> {
            if range.end - range.start > 1 {
                let partition = (range.end + range.start) / 2;
                let (mut a, mut b) = rayon::join(
                    || search_docs(s, re, range.start..partition),
                    || search_docs(s, re, partition..range.end),
                );
                a.append(&mut b);
                return a;
            }

            let doc_id = range.start;
            let matched_ranges = re
                .find_iter(s.doc_content(doc_id))
                .map(|m| m.start() as u32..m.end() as u32)
                .collect::<Vec<Range<u32>>>();
            if matched_ranges.len() > 0 {
                vec![DocMatches {
                    doc: doc_id,
                    matched_ranges,
                }]
            } else {
                Vec::new()
            }
        };
        let num_docs = self.header.doc_starts_len as u32;
        search_docs(self, &re, 0..num_docs)
    }
}

pub struct DocMatches {
    pub doc: DocID,
    pub matched_ranges: Vec<Range<u32>>,
}

pub struct DocumentIterator<'a> {
    shard: &'a Shard,
    next_id: DocID,
}

impl<'a> DocumentIterator<'a> {
    fn new(shard: &'a Shard) -> Self {
        Self { shard, next_id: 0 }
    }
}

impl<'a> Iterator for DocumentIterator<'a> {
    type Item = (DocID, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_id >= self.shard.header.doc_starts_len as u32 {
            None
        } else {
            let res = Some((self.next_id, self.shard.doc_content(self.next_id as u32)));
            self.next_id += 1;
            res
        }
    }
}

#[repr(C)]
#[derive(Debug)]
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

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::HEADER_SIZE as usize);
        buf.write(&self.version.to_le_bytes());
        buf.write(&self.flags.to_le_bytes());
        buf.write(&self.id.to_le_bytes());
        buf.write(&self.content_ptr.to_le_bytes());
        dbg!(&self.content_ptr);
        buf.write(&self.content_len.to_le_bytes());
        buf.write(&self.doc_starts_ptr.to_le_bytes());
        buf.write(&self.doc_starts_len.to_le_bytes());
        buf.write(&self.sa_ptr.to_le_bytes());
        buf.write(&self.sa_len.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut s = Self::default();
        s.version = u16::from_le_bytes(buf[0..2].try_into()?);
        s.flags = u16::from_le_bytes(buf[2..4].try_into()?);
        s.id = u32::from_le_bytes(buf[4..8].try_into()?);
        s.content_ptr = u64::from_le_bytes(buf[8..16].try_into()?);
        s.content_len = u64::from_le_bytes(buf[16..24].try_into()?);
        s.doc_starts_ptr = u64::from_le_bytes(buf[24..32].try_into()?);
        s.doc_starts_len = u64::from_le_bytes(buf[32..40].try_into()?);
        s.sa_ptr = u64::from_le_bytes(buf[40..48].try_into()?);
        s.sa_len = u64::from_le_bytes(buf[48..56].try_into()?);
        Ok(s)
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
