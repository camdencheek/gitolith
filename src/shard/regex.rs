use itertools::{Itertools, Product};
use regex_syntax::hir::{self, visit, Hir, HirKind, Visitor};
use std::ops::{Range, RangeInclusive};

pub struct RangesBuilder {
    open: Option<SuffixRangeIter>,
    closed: Vec<SuffixRangeIter>,
}

impl RangesBuilder {
    pub fn new() -> Self {
        Self {
            open: None,
            closed: Vec::new(),
        }
    }

    pub fn from_hir(hir: Hir) -> Self {
        let mut s = Self::new();
        s.push_hir(hir);
        s
    }

    pub fn build(mut self) -> Vec<SuffixRangeIter> {
        if let Some(open) = self.open {
            self.closed.push(open)
        }
        self.closed
    }

    pub fn push_hir(&mut self, hir: Hir) {
        use hir::HirKind::*;

        match hir.into_kind() {
            Empty => {}
            Literal(lit) => self.push_literal(lit),
            Class(class) => self.push_class(class),
            Group(g) => self.push_hir(*g.hir),
            Concat(hirs) => self.push_concat(hirs),
            Alternation(hirs) => self.push_alternation(hirs),
            // Everything below this line does not help
            // narrow the search. TODO: see if we can maintain
            // some structure here so we can use these to filter
            // after we've generated suffix ranges.
            Anchor(a) => self.push_anchor(a),
            WordBoundary(wb) => self.push_word_boundary(wb),
            Repetition(rep) => self.push_repetition(rep),
        }
    }

    fn close_opened(&mut self) {
        if let Some(open) = self.open.take() {
            self.closed.push(open);
            self.open = None;
        }
    }

    fn get_open_mut(&mut self) -> &mut SuffixRangeIter {
        if self.open.is_none() {
            self.open = Some(SuffixRangeIter::Empty(EmptyIterator::new()));
        }

        self.open
            .as_mut()
            .expect("just set the option, should be not be none")
    }

    fn push_concat(&mut self, hirs: Vec<Hir>) {
        for child in hirs.into_iter() {
            self.push_hir(child);
        }
    }

    fn push_literal(&mut self, lit: hir::Literal) {
        use hir::Literal;
        let open = self.get_open_mut();

        // TODO it feels like I shouldn't need the clones here.
        *open = match lit {
            Literal::Byte(b) => {
                SuffixRangeIter::ByteLiteral(ByteLiteralAppender::new(open.clone(), b))
            }
            Literal::Unicode(char) => {
                SuffixRangeIter::UnicodeLiteral(UnicodeLiteralAppender::new(open.clone(), char))
            }
        }
    }

    fn push_class(&mut self, class: hir::Class) {
        use hir::Class;

        let open = self.get_open_mut();

        // TODO it feels like I shouldn't need the clones here.
        *open = match class {
            Class::Bytes(ref bc) => {
                SuffixRangeIter::ByteClass(ByteClassAppender::new(open.clone(), bc.clone()))
            }
            Class::Unicode(ref uc) => {
                SuffixRangeIter::UnicodeClass(UnicodeClassAppender::new(open.clone(), uc.clone()))
            }
        }
    }

    fn push_alternation(&mut self, alts: Vec<Hir>) {
        // Alternation is tricky. If the alternation is all literals/classes, we can handle this
        // efficiently, but if if there are anchors or repetitions, it becomes much more difficult.
        // We take the easy path and only handle alternations with simple children.

        for alt in &alts {
            let v = HirVisitor {};
            if visit(alt, v).is_err() {
                self.close_opened();
                return;
            };
        }

        // If we got here, all children are simple patterns.
        let mut alt_iters = Vec::with_capacity(alts.len());
        for alt in alts {
            // Cheat and reuse machinery from the builder
            let mut built = Self::from_hir(alt).build();
            debug_assert!(built.len() == 1);
            alt_iters.push(built.pop().unwrap());
        }

        let open = self.get_open_mut();
        *open = SuffixRangeIter::Alternation(AlternationAppender::new(open.clone(), alt_iters))
    }

    fn push_anchor(&mut self, anchor: hir::Anchor) {
        self.close_opened()
    }

    fn push_word_boundary(&mut self, wb: hir::WordBoundary) {
        self.close_opened()
    }

    fn push_repetition(&mut self, rep: hir::Repetition) {
        self.close_opened()
    }
}

struct HirVisitor {}

impl Visitor for HirVisitor {
    type Output = ();
    type Err = ();

    fn finish(self) -> Result<Self::Output, Self::Err> {
        Ok(())
    }

    fn visit_pre(&mut self, hir: &Hir) -> Result<(), Self::Err> {
        use hir::HirKind::*;

        match hir.kind() {
            Anchor(_) | WordBoundary(_) | Repetition(_) => Err(()),
            _ => Ok(()),
        }
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

#[derive(Clone)]
pub enum SuffixRangeIter {
    Empty(EmptyIterator),
    ByteLiteral(ByteLiteralAppender),
    UnicodeLiteral(UnicodeLiteralAppender),
    ByteClass(ByteClassAppender),
    UnicodeClass(UnicodeClassAppender),
    Alternation(AlternationAppender),
}

impl Iterator for SuffixRangeIter {
    type Item = LitRange;

    fn next(&mut self) -> Option<Self::Item> {
        use SuffixRangeIter::*;
        match self {
            Empty(v) => v.next(),
            ByteLiteral(v) => v.next(),
            UnicodeLiteral(v) => v.next(),
            ByteClass(v) => v.next(),
            UnicodeClass(v) => v.next(),
            Alternation(v) => v.next(),
        }
    }
}

impl SuffixRangeIterator for SuffixRangeIter {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        use SuffixRangeIter::*;
        match self {
            Empty(v) => v.depth_hint(),
            ByteLiteral(v) => v.depth_hint(),
            UnicodeLiteral(v) => v.depth_hint(),
            ByteClass(v) => v.depth_hint(),
            UnicodeClass(v) => v.depth_hint(),
            Alternation(v) => v.depth_hint(),
        }
    }
}

#[derive(Clone)]
pub struct EmptyIterator(std::iter::Once<LitRange>);

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
pub struct ByteLiteralAppender {
    predecessor: Box<SuffixRangeIter>,
    byte: u8,
}

impl ByteLiteralAppender {
    fn new(predecessor: SuffixRangeIter, byte: u8) -> Self {
        Self {
            predecessor: Box::new(predecessor),
            byte,
        }
    }
}

impl Iterator for ByteLiteralAppender {
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

impl SuffixRangeIterator for ByteLiteralAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.predecessor.depth_hint();
        (low + 1, high.map(|i| i + 1))
    }
}

#[derive(Clone)]
pub struct UnicodeLiteralAppender {
    predecessor: Box<SuffixRangeIter>,
    char: char,
}

impl UnicodeLiteralAppender {
    fn new(pred: SuffixRangeIter, char: char) -> Self {
        Self {
            predecessor: Box::new(pred),
            char,
        }
    }
}

impl Iterator for UnicodeLiteralAppender {
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

impl SuffixRangeIterator for UnicodeLiteralAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        let char_size = self.char.len_utf8();
        let (low, high) = self.predecessor.depth_hint();
        (low + char_size, high.map(|i| i + char_size))
    }
}

#[derive(Clone)]
pub struct ByteClassAppender {
    product: Box<Product<SuffixRangeIter, std::vec::IntoIter<hir::ClassBytesRange>>>,
    depth_hint: (usize, Option<usize>),
}

impl ByteClassAppender {
    pub fn new(predecessor: SuffixRangeIter, class: hir::ClassBytes) -> Self {
        let (depth_low, depth_high) = predecessor.depth_hint();
        Self {
            product: Box::new(predecessor.cartesian_product(class.ranges().to_vec())),
            depth_hint: (depth_low + 1, depth_high.map(|i| i + 1)),
        }
    }
}

impl Iterator for ByteClassAppender {
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

impl SuffixRangeIterator for ByteClassAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        self.depth_hint
    }
}

#[derive(Clone)]
pub struct UnicodeClassAppender {
    product: Box<Product<SuffixRangeIter, UnicodeRangeSplitIterator>>,
    depth_hint: (usize, Option<usize>),
}

impl UnicodeClassAppender {
    pub fn new(predecessor: SuffixRangeIter, class: hir::ClassUnicode) -> Self {
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
            product: Box::new(predecessor.cartesian_product(UnicodeRangeSplitIterator::new(class))),
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
pub struct AlternationAppender {
    product: Box<
        itertools::Product<
            SuffixRangeIter,
            std::iter::Flatten<std::vec::IntoIter<SuffixRangeIter>>,
        >,
    >,
    depth_hint: (usize, Option<usize>),
}

impl AlternationAppender {
    fn new(pred: SuffixRangeIter, alts: Vec<SuffixRangeIter>) -> Self {
        assert!(!alts.is_empty());
        let (base_low, base_high) = pred.depth_hint();
        let depth_hint = alts.iter().fold((usize::MAX, Some(0)), |acc, it| {
            let (l, h) = it.depth_hint();
            let new_min = usize::min(acc.0, l);
            let new_max = match (acc.1, h) {
                (Some(i), Some(j)) => Some(usize::max(i, j)),
                _ => None,
            };
            (new_min, new_max)
        });
        Self {
            product: Box::new(pred.cartesian_product(alts.into_iter().flatten())),
            depth_hint,
        }
    }
}

impl Iterator for AlternationAppender {
    type Item = LitRange;

    fn next(&mut self) -> Option<Self::Item> {
        let (base, addition) = self.product.next()?;
        let (mut start, mut end) = base.into_inner();
        start.extend_from_slice(addition.start());
        end.extend_from_slice(addition.end());
        Some(start..=end)
    }
}

impl SuffixRangeIterator for AlternationAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        self.depth_hint
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
