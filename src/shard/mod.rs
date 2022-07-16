use memmap::{Mmap, MmapMut};
mod regex;
use itertools::{Itertools, Product};
use regex_syntax::hir::{self, Hir, HirKind};
use std::fs::File;
use std::io::{self, Error, Read, Seek, SeekFrom, Write};
use std::iter::Iterator;
use std::ops::{Range, RangeInclusive};
use std::os::unix::fs::FileExt;
use std::path::Path;
use suffix;

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

        let mut stypes = suffix::SuffixTypes::new(sa.len() as u32);
        let mut bins = suffix::Bins::new();
        suffix::sais(sa, &mut stypes, &mut bins, &suffix::Utf8(content));

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

    fn content(&self) -> &[u8] {
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

    pub fn sa_range(&self, r: Range<&[u8]>) -> &[u32] {
        assert!(r.start < r.end);
        &self.sa()[self.sa_find_start(r.start) as usize..self.sa_find_start(r.end) as usize]
    }

    // finds the index of the first suffix whose prefix is needle
    pub fn sa_find_start(&self, needle: &[u8]) -> u32 {
        let sa = self.sa();
        let content = self.content();
        let (mut low, mut high) = (0usize, sa.len() - 1);
        while low < high {
            let mid = (low + high) / 2;
            let suffix = &content[sa[mid] as usize..];
            if suffix >= needle {
                high = mid
            } else {
                low = mid + 1
            }
        }
        low as u32
    }

    // finds the index of the first suffix whose prefix is greater than needle
    pub fn sa_find_end(&self, needle: &[u8]) -> u32 {
        let sa = self.sa();
        let content = self.content();
        let (mut low, mut high) = (0usize, sa.len() - 1);
        while low < high {
            let mid = (low + high) / 2;
            let suffix = &content[sa[mid] as usize..sa[mid] as usize + needle.len()];
            if suffix > needle {
                high = mid
            } else {
                low = mid + 1
            }
        }
        low as u32
    }
}

struct RegexMatchIterator {}

impl RegexMatchIterator {
    fn new(ir: hir::Hir) -> Self {
        match ir.kind() {}
    }
}

// A range of byte string literals.
// Invariant: start <= end.
type LitRange = RangeInclusive<Vec<u8>>;

trait SuffixRangeIterator: Iterator<Item = LitRange> {
    // A lower and optional upper bound on the length of the byte vec bounds
    // on the yielded LitRange.
    fn depth_hint(&self) -> (usize, Option<usize>);
}

enum SuffixRangeIter<T: SuffixRangeIterator> {
    Empty(EmptyIterator),
    ByteLiteral(ByteLiteralAppender<T>),
    UnicodeLiteral(UnicodeLiteralAppender<T>),
    ByteClass(ByteClassAppender<T>),
    UnicodeClass(UnicodeClassAppender<T>),
    Alternation(AlternationIterator<T>),
}

#[derive(Clone)]
struct EmptyIterator(std::iter::Once<LitRange>);

impl EmptyIterator {
    fn new() -> Self {
        Self(std::iter::once(b"".to_vec()..=b"".to_vec()))
    }
}

impl Iterator for EmptyIterator {
    type Item = LitRange;

    fn next(&mut self) -> Option<LitRange> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl SuffixRangeIterator for EmptyIterator {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

#[derive(Clone)]
struct ByteLiteralAppender<T>
where
    T: SuffixRangeIterator,
{
    predecessor: T,
    byte: u8,
}

impl<T> Iterator for ByteLiteralAppender<T>
where
    T: SuffixRangeIterator,
{
    type Item = LitRange;

    fn next(&mut self) -> Option<LitRange> {
        match self.predecessor.next() {
            Some(r) => {
                let (mut start, mut end) = r.into_inner();
                start.push(self.byte);
                end.push(self.byte);
                Some(start..=end)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.predecessor.size_hint()
    }
}

impl<T> SuffixRangeIterator for ByteLiteralAppender<T>
where
    T: SuffixRangeIterator,
{
    fn depth_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.predecessor.depth_hint();
        (low + 1, high.map(|i| i + 1))
    }
}

#[derive(Clone)]
struct UnicodeLiteralAppender<T>
where
    T: SuffixRangeIterator,
{
    predecessor: T,
    char: char,
}

impl<T> Iterator for UnicodeLiteralAppender<T>
where
    T: SuffixRangeIterator,
{
    type Item = LitRange;

    fn next(&mut self) -> Option<LitRange> {
        match self.predecessor.next() {
            Some(r) => {
                let mut buf = [0u8; 4];
                let bytes = self.char.encode_utf8(&mut buf[..]).as_bytes();
                let (mut start, mut end) = r.into_inner();
                start.extend_from_slice(bytes);
                end.extend_from_slice(bytes);
                Some(start..=end)
            }
            None => None,
        }
    }
}

impl<T> SuffixRangeIterator for UnicodeLiteralAppender<T>
where
    T: SuffixRangeIterator,
{
    fn depth_hint(&self) -> (usize, Option<usize>) {
        let char_size = self.char.len_utf8();
        let (low, high) = self.predecessor.depth_hint();
        (low + char_size, high.map(|i| i + char_size))
    }
}

#[derive(Clone)]
struct ByteClassAppender<T>
where
    T: SuffixRangeIterator,
{
    product: Product<T, std::vec::IntoIter<hir::ClassBytesRange>>,
    depth_hint: (usize, Option<usize>),
}

impl<T> ByteClassAppender<T>
where
    T: SuffixRangeIterator,
{
    pub fn new(predecessor: T, class: hir::ClassBytes) -> Self {
        let (depth_low, depth_high) = predecessor.depth_hint();
        Self {
            product: predecessor.cartesian_product(class.ranges().to_vec()),
            depth_hint: (depth_low + 1, depth_high.map(|i| i + 1)),
        }
    }
}

impl<T> Iterator for ByteClassAppender<T>
where
    T: SuffixRangeIterator,
{
    type Item = LitRange;

    fn next(&mut self) -> Option<LitRange> {
        let (curr, range) = self.product.next()?;
        let (mut start, mut end) = curr.into_inner();
        start.push(range.start());
        end.push(range.end());
        Some(start..=end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.product.size_hint()
    }
}

impl<T> SuffixRangeIterator for ByteClassAppender<T>
where
    T: SuffixRangeIterator,
{
    fn depth_hint(&self) -> (usize, Option<usize>) {
        self.depth_hint
    }
}

#[derive(Clone)]
struct UnicodeClassAppender<T>
where
    T: SuffixRangeIterator,
{
    product: Product<T, UnicodeRangeSplitIterator>,
    depth_hint: (usize, Option<usize>),
}

impl<T> UnicodeClassAppender<T>
where
    T: SuffixRangeIterator,
{
    pub fn new(predecessor: T, class: hir::ClassUnicode) -> Self {
        let (depth_low, depth_high) = predecessor.depth_hint();
        let min_char_len = class
            .ranges()
            .first()
            .expect("rangess should never be empty")
            .start()
            .len_utf8();
        let max_char_len = class
            .ranges()
            .last()
            .expect("ranges should never be empty")
            .end()
            .len_utf8();
        Self {
            product: predecessor.cartesian_product(UnicodeRangeSplitIterator::new(class)),
            depth_hint: (
                depth_low + min_char_len,
                depth_high.map(|i| i + max_char_len),
            ),
        }
    }
}

impl Iterator for UnicodeClassAppender {
    type Item = LitRange;

    fn next(&mut self) -> Option<LitRange> {
        let (lit_range, unicode_range) = self.product.next()?;
        let (mut start, mut end) = lit_range.into_inner();
        let mut buf = [0u8; 4];
        start.extend_from_slice(unicode_range.start().encode_utf8(&mut buf).as_bytes());
        end.extend_from_slice(unicode_range.end().encode_utf8(&mut buf).as_bytes());
        Some(start..=end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.product.size_hint()
    }
}

impl SuffixRangeIterator for UnicodeClassAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        self.depth_hint
    }
}

#[derive(Clone)]
struct AlternationIterator<T>
where
    T: SuffixRangeIterator,
{
    predecessor: T,
    alternatives: U,
}

impl<T, U> AlternationIterator<T, U>
where
    T: SuffixRangeIterator,
    U: Iterator<Item = Box<dyn SuffixRangeIterator>> + Clone,
{
    fn new(predecessor: T, alternatives: U) -> Self {
        Self {
            predecessor,
            alternatives,
        }
    }
}

#[derive(Clone)]
struct UnicodeRangeSplitIterator {
    product: Product<
        std::vec::IntoIter<hir::ClassUnicodeRange>,
        std::array::IntoIter<hir::ClassUnicodeRange, 4>,
    >,
}

impl UnicodeRangeSplitIterator {
    fn new(class: hir::ClassUnicode) -> Self {
        let new_range = |a: u32, b: u32| {
            hir::ClassUnicodeRange::new(char::from_u32(a).unwrap(), char::from_u32(b).unwrap())
        };
        // TODO this could probably be const
        let sized_ranges = [
            new_range(0, 0x007F),
            new_range(0x0080, 0x07FF),
            new_range(0x0800, 0xFFFF),
            new_range(0x10000, 0x10FFFF),
        ];
        Self {
            product: class
                .ranges()
                .to_vec()
                .into_iter()
                .cartesian_product(sized_ranges.into_iter()),
        }
    }

    fn intersect(
        left: hir::ClassUnicodeRange,
        right: hir::ClassUnicodeRange,
    ) -> Option<hir::ClassUnicodeRange> {
        let start = char::max(left.start(), right.start());
        let end = char::min(left.end(), right.end());
        if start <= end {
            Some(hir::ClassUnicodeRange::new(start, end))
        } else {
            None
        }
    }
}

impl Iterator for UnicodeRangeSplitIterator {
    type Item = hir::ClassUnicodeRange;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (left, right) = self.product.next()?;
            if let Some(range) = Self::intersect(left, right) {
                return Some(range);
            }
        }
    }
}

struct DocumentIterator<'a> {
    shard: &'a Shard,
    next_id: DocID,
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
