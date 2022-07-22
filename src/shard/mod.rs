mod regex;
mod suffix;

pub use self::regex::*;
use self::suffix::SuffixIdx;
use ::regex::bytes::Regex;
use ::suffix as suf;
use memmap::{Mmap, MmapMut};
use rayon;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::iter::Iterator;
use std::ops::{Range, RangeInclusive};
use std::os::unix::fs::FileExt;
use std::path::Path;
mod docs;
use docs::{Doc, DocSlice};

pub type ShardID = u32;
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
        f.seek(SeekFrom::Start(ShardHeader::HEADER_SIZE as u64))?;

        // Content starts immediately after the header
        header.content_ptr = ShardHeader::HEADER_SIZE as u64;

        let mut current_doc_start: u32 = 0;
        let mut doc_starts: Vec<u32> = Vec::with_capacity(docs.size_hint().0);

        let zero_byte: [u8; 1] = [0; 1];
        for mut doc in docs {
            let doc_len = io::copy(&mut doc, &mut f)?;
            if header.content_len + doc_len + 1 >= u32::MAX.into() {
                // truncate file back to the length before hitting the limit
                f.set_len(header.content_ptr + current_doc_start as u64)?;
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
        f.set_len(file_size)?;

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

    pub fn open(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let f = File::open(path)?;
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

    pub fn docs(&self) -> DocSlice<'_> {
        DocSlice::new(0, self.doc_starts(), self.content())
    }

    // returns a slice of all prefixes that start with the literal needle
    pub fn sa_prefixes(&self, needle: &[u8]) -> &[u32] {
        &self.sa()[self.sa_find_start(needle) as usize..self.sa_find_end(needle) as usize]
    }

    pub fn sa_slice<T>(&self, r: RangeInclusive<T>) -> &[SuffixIdx]
    where
        T: AsRef<[u8]> + Ord,
    {
        debug_assert!(r.start() <= r.end());
        &self.sa()[self.sa_find_start(r.start().as_ref()) as usize
            ..self.sa_find_end(r.end().as_ref()) as usize]
    }

    pub fn sa_range<T>(&self, r: RangeInclusive<T>) -> Range<SuffixIdx>
    where
        T: AsRef<[u8]> + Ord,
    {
        debug_assert!(r.start() <= r.end());
        self.sa_find_start(r.start().as_ref())..self.sa_find_end(r.end().as_ref())
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

    pub fn search_skip_index(&self, re: Regex) -> Vec<DocMatches> {
        // Quick and dirty work-stealing with rayon.
        // TODO would be nice if this returned deterministic results.
        // TODO would be nice if this iterated over results rather than collected them.
        // TODO make this respect a limit.
        fn search_docs<'a>(re: &Regex, docs: DocSlice<'a>) -> Vec<DocMatches<'a>> {
            if docs.len() > 1 {
                let partition = docs.len() / 2;
                let (mut a, mut b) = rayon::join(
                    || search_docs(re, docs.index(..partition)),
                    || search_docs(re, docs.index(partition..)),
                );
                a.append(&mut b);
                return a;
            }
            let doc = docs.index(0);
            let matched_ranges = re
                .find_iter(doc.content)
                .map(|m| m.start() as u32..m.end() as u32)
                .collect::<Vec<Range<u32>>>();
            if matched_ranges.len() > 0 {
                vec![DocMatches {
                    doc,
                    matched_ranges,
                }]
            } else {
                Vec::new()
            }
        }
        search_docs(&re, self.docs())
    }

    pub fn search<'a, 'b: 'a>(
        &'a self,
        re: &'b Regex,
    ) -> Box<dyn Iterator<Item = regex::DocMatches<'a>> + 'a> {
        regex::new_regex_iter(self, &re)
    }
}

pub struct DocMatches<'a> {
    pub doc: Doc<'a>,
    pub matched_ranges: Vec<Range<u32>>,
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

    // TODO add a first character index
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::HEADER_SIZE as usize);
        buf.write(&self.version.to_le_bytes()).unwrap();
        buf.write(&self.flags.to_le_bytes()).unwrap();
        buf.write(&self.id.to_le_bytes()).unwrap();
        buf.write(&self.content_ptr.to_le_bytes()).unwrap();
        dbg!(&self.content_ptr);
        buf.write(&self.content_len.to_le_bytes()).unwrap();
        buf.write(&self.doc_starts_ptr.to_le_bytes()).unwrap();
        buf.write(&self.doc_starts_len.to_le_bytes()).unwrap();
        buf.write(&self.sa_ptr.to_le_bytes()).unwrap();
        buf.write(&self.sa_len.to_le_bytes()).unwrap();
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

#[cfg(test)]
mod test {
    use super::*;
    use ::regex::bytes::Regex;
    use std::error::Error;
    use std::io::Cursor;

    fn corpus(strs: Vec<&str>) -> impl Iterator<Item = Cursor<&str>> {
        strs.into_iter().map(|s| Cursor::new(s))
    }

    fn do_search(s: &Shard, re: &str) -> String {
        let re = Regex::new(re).unwrap();
        let mut buf = String::new();
        for m in s.search(&re) {
            buf.push_str(&format!("DocID: {}\n", m.doc.id));
            for range in m.matches {
                buf.push_str(
                    std::str::from_utf8(&m.doc.content[range.start as usize..range.end as usize])
                        .unwrap(),
                );
                buf.push_str("\n");
            }
        }
        buf
    }

    #[test]
    fn simple() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def"]))?;
        let res = do_search(&s, "bc");
        assert_eq!(
            res,
            "DocID: 0\n\
            bc\n",
        );
        Ok(())
    }

    #[test]
    fn case_sensitive() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def"]))?;
        let res = do_search(&s, "(?i:BC)");
        assert_eq!(
            res,
            "DocID: 0\n\
            bc\n",
        );
        Ok(())
    }

    #[test]
    fn cross_doc_boundaries() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def"]))?;
        let res = do_search(&s, "bc\0d");
        assert_eq!(res, "",);
        Ok(())
    }

    #[test]
    fn simple_hole() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def"]))?;
        let res = do_search(&s, "a.c");
        assert_eq!(
            res,
            "DocID: 0\n\
            abc\n",
        );
        Ok(())
    }

    #[test]
    fn hole_cross_docs() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def"]))?;
        let res = do_search(&s, "a.*f");
        assert_eq!(res, "",);
        Ok(())
    }

    #[test]
    fn hole_cross_docs_reverse() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def"]))?;
        let res = do_search(&s, "f.*a");
        assert_eq!(res, "",);
        Ok(())
    }

    #[test]
    fn classes() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def", "ghi"]))?;
        let res = do_search(&s, r"\w+");
        assert_eq!(
            res,
            "DocID: 0\n\
            abc\n\
            DocID: 1\n\
            def\n\
            DocID: 2\n\
            ghi\n",
        );
        Ok(())
    }

    #[test]
    fn multi_match_docs() -> Result<(), Box<dyn Error>> {
        let f = tempfile::NamedTempFile::new()?;
        let s = Shard::new(0, f.path(), corpus(vec!["abc", "def"]))?;
        let res = do_search(&s, r"\w");
        assert_eq!(
            res,
            "DocID: 0\n\
            a\n\
            b\n\
            c\n\
            DocID: 1\n\
            d\n\
            e\n\
            f\n",
        );
        Ok(())
    }
}
