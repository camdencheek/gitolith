use itertools::{Itertools, Product};
use regex_syntax::hir::{self, Hir};
use std::ops::RangeInclusive;

// A range of byte string literals, intended to be used as
// Invariant: start <= end.
type PrefixRange = RangeInclusive<Vec<u8>>;

trait PrefixRangeIterExt: Iterator<Item = PrefixRange> {
    // A lower and upper bound on the length of the byte vec bounds
    // on the yielded PrefixRanges.
    fn prefix_len_bounds(&self) -> (usize, usize);
}

enum ExtractedRegexLiterals {
    // An exact extraction indicates that every prefix yielded by the iterator is an exact match to
    // the input regex pattern and does not need to be rechecked with the original regex pattern.
    Exact(PrefixRangeIter),

    // An inexact extraction requires any matched document to be rechecked. It guarantees that the
    // only documents that can possibly match the original regex query will contain at least one of
    // the prefixes from each of the iterators.
    //
    // As an example, the regex query /ab(c|d).*ef(g|h)/ will yield something like
    // Inexact(vec![[abc, abd], [efg, efh]])
    Inexact(Vec<PrefixRangeIter>),

    // If no meaningful literals can be extracted from the regex pattern, like /.*/,
    // the result of extraction will be None.
    None,
}

impl ExtractedRegexLiterals {
    pub fn optimize(&mut self) {
        *self = match self {
            Self::None => Self::None,
            Self::Exact(p) => Self::optimize_exact(p),
            Self::Inexact(ps) => Self::optimize_inexact(ps),
        }
    }

    fn optimize_exact(pri: PrefixRangeIter) -> Self {
        // Unless we get at least 2 bytes worth of data to work with
        // it's probably not worth it, so just fall back to unindexed search.
        // TODO: tune this heuristic
        if pri.prefix_len_bounds() < 2 {
            return self::None;
        }
    }

    fn optimize_inexact(pris: Vec<PrefixRangeIter>) -> Self {
        // Sort by minimum prefix size descending
        pris.sort_by_key(|pri| std::cmp::Reverse(pri.prefix_len_bounds().0));
        // Only keep the 3 longest prefixes
        pris.truncate(3);
    }

    fn limit_len(pri: PrefixRangeIter) -> (PrefixRangeIter, bool) {
        // TODO: tune this number
        if pri.len() < 1024 {
            // No need to truncate, this is a reasonable size to iterate over
            return (pri, false);
        }

        let (pred, _) = Self::limit_len(pri.predecessor());
        (pred, true)
    }
}

struct RegexLiteralExtractor {
    // An iterator over the current set of prefixes being built, or
    // None if no prefix set has been started.
    current: Option<PrefixRangeIter>,

    // A set of prefix ranges that have already been extended to their
    // maximum length given a pattern.
    complete: Vec<PrefixRangeIter>,

    exact: bool,
}

impl RegexLiteralExtractor {
    pub fn from_hir(hir: Hir) -> ExtractedRegexLiterals {
        let mut s = Self {
            current: None,
            complete: Vec::new(),
            exact: true,
        };
        s.push_hir(hir);

        match (self.current, exact) {
            (Some(c), true) => ExtractedRegexLiterals::Exact(c),
            (None, true) => ExtractedRegexLiterals::None,
            (Some(c), false) => {
                self.complete.push(c);
                ExtractedRegexLiterals::Inexact(self.complete)
            }
            (None, false) => {
                if self.complete.len() == 0 {
                    ExtractedRegexLiterals::None
                } else {
                    ExtractedRegexLiterals::Inexact(self.complete)
                }
            }
        }
    }

    fn push_hir(&mut self, hir: Hir) {
        use regex_syntax::hir::HirKind::*;

        match hir.into_kind() {
            Empty => {}
            Literal(lit) => self.push_literal(lit),
            Class(class) => self.push_class(class),
            Group(g) => self.push_hir(*g.hir),
            Concat(hirs) => self.push_concat(hirs),
            Alternation(hirs) => self.push_alternation(hirs),
            // Everything below this does not help narrow the search.
            // TODO: we may be able to get some value from "one or more" type repetitions.
            Anchor(a) => self.push_anchor(a),
            WordBoundary(wb) => self.push_word_boundary(wb),
            Repetition(rep) => self.push_repetition(rep),
        }
    }

    fn close_current(&mut self) {
        if let Some(open) = self.current.take() {
            self.complete.push(open);
            self.current = None;
        }
    }

    // cannot_handle marks the current set of prefixes as complete
    // and marks the document iterator as inexact.
    fn cannot_handle(&mut self) {
        self.close_current();
        self.exact = false;
    }

    fn get_open_mut(&mut self) -> &mut PrefixRangeIter {
        if self.current.is_none() {
            self.current = Some(PrefixRangeIter::Empty(EmptyIterator::new()));
        }

        self.current
            .as_mut()
            .expect("just set the option, should be not be none")
    }

    fn push_concat(&mut self, hirs: Vec<Hir>) {
        for child in hirs.into_iter() {
            if let Some(c) = &self.current {
                // Arbitrary limit to avoid exponential growth in the number
                // of ranges generated by case insensitive search
                // TODO: do this in a pruning step after generation.
                if c.len() > 1024 {
                    self.exact = false;
                    return;
                }
            }
            self.push_hir(child);
        }
    }

    fn push_literal(&mut self, lit: hir::Literal) {
        use hir::Literal;
        let open = self.get_open_mut();

        // TODO it feels like I shouldn't need the clones here.
        *open = match lit {
            Literal::Byte(b) => {
                PrefixRangeIter::ByteLiteral(ByteLiteralAppender::new(open.clone(), b))
            }
            Literal::Unicode(char) => {
                PrefixRangeIter::UnicodeLiteral(UnicodeLiteralAppender::new(open.clone(), char))
            }
        }
    }

    fn push_class(&mut self, class: hir::Class) {
        use hir::Class;

        let open = self.get_open_mut();

        // TODO it feels like I shouldn't need the clones here.
        *open = match class {
            Class::Bytes(ref bc) => {
                PrefixRangeIter::ByteClass(ByteClassAppender::new(open.clone(), bc.clone()))
            }
            Class::Unicode(ref uc) => {
                PrefixRangeIter::UnicodeClass(UnicodeClassAppender::new(open.clone(), uc.clone()))
            }
        }
    }

    fn push_alternation(&mut self, alts: Vec<Hir>) {
        // Alternation is tricky. If the alternation is all literals/classes, we can handle this
        // efficiently, but if if there are anchors or repetitions, it becomes much more difficult.
        // We take the easy path and only handle alternations with simple children.

        // A method-scoped type that is just used to visit each node of the Hir
        // and report whether there are any Anchors, WordBoundaries, or Repetitions.
        struct SimpleHirChecker {}

        impl hir::Visitor for SimpleHirChecker {
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

        for alt in &alts {
            if hir::visit(alt, SimpleHirChecker {}).is_err() {
                self.cannot_handle();
                return;
            };
        }

        // If we got here, all children are simple patterns.
        let mut alt_iters = Vec::with_capacity(alts.len());
        for alt in alts {
            let mut built = Self::from_hir(alt);
            debug_assert!(built.len() == 1);
            alt_iters.push(built.pop().unwrap());
        }

        let open = self.get_open_mut();
        *open = PrefixRangeIter::Alternation(AlternationAppender::new(open.clone(), alt_iters))
    }

    fn push_anchor(&mut self, _anchor: hir::Anchor) {
        self.cannot_handle();
    }

    fn push_word_boundary(&mut self, _wb: hir::WordBoundary) {
        self.cannot_handle();
    }

    fn push_repetition(&mut self, _rep: hir::Repetition) {
        self.cannot_handle();
    }
}

#[derive(Clone)]
pub enum PrefixRangeIter {
    Empty(EmptyIterator),
    ByteLiteral(ByteLiteralAppender),
    UnicodeLiteral(UnicodeLiteralAppender),
    ByteClass(ByteClassAppender),
    UnicodeClass(UnicodeClassAppender),
    Alternation(AlternationAppender),
}

impl Iterator for PrefixRangeIter {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<Self::Item> {
        use PrefixRangeIter::*;
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

impl ExactSizeIterator for PrefixRangeIter {
    fn len(&self) -> usize {
        use PrefixRangeIter::*;
        match self {
            Empty(v) => v.len(),
            ByteLiteral(v) => v.len(),
            UnicodeLiteral(v) => v.len(),
            ByteClass(v) => v.len(),
            UnicodeClass(v) => v.len(),
            Alternation(v) => v.len(),
        }
    }
}

impl PrefixRangeIterExt for PrefixRangeIter {
    fn prefix_len_bounds(&self) -> (usize, usize) {
        use PrefixRangeIter::*;
        match self {
            Empty(v) => v.prefix_len_bounds(),
            ByteLiteral(v) => v.prefix_len_bounds(),
            UnicodeLiteral(v) => v.prefix_len_bounds(),
            ByteClass(v) => v.prefix_len_bounds(),
            UnicodeClass(v) => v.prefix_len_bounds(),
            Alternation(v) => v.prefix_len_bounds(),
        }
    }
}

#[derive(Clone)]
pub struct EmptyIterator(std::iter::Once<PrefixRange>);

impl EmptyIterator {
    fn new() -> Self {
        Self(std::iter::once(b"".to_vec()..=b"".to_vec()))
    }
}

impl Iterator for EmptyIterator {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        self.0.next()
    }
}

impl ExactSizeIterator for EmptyIterator {
    fn len(&self) -> usize {
        1
    }
}

impl PrefixRangeIterExt for EmptyIterator {
    fn prefix_len_bounds(&self) -> (usize, usize) {
        (0, 0)
    }
}

#[derive(Clone)]
pub struct ByteLiteralAppender {
    predecessor: Box<PrefixRangeIter>,
    byte: u8,
}

impl ByteLiteralAppender {
    fn new(predecessor: PrefixRangeIter, byte: u8) -> Self {
        Self {
            predecessor: Box::new(predecessor),
            byte,
        }
    }
}

impl Iterator for ByteLiteralAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
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
}

impl ExactSizeIterator for ByteLiteralAppender {
    fn len(&self) -> usize {
        self.predecessor.len()
    }
}

impl PrefixRangeIterExt for ByteLiteralAppender {
    fn prefix_len_bounds(&self) -> (usize, usize) {
        let (low, high) = self.predecessor.prefix_len_bounds();
        (low + 1, high + 1)
    }
}

#[derive(Clone)]
pub struct UnicodeLiteralAppender {
    predecessor: Box<PrefixRangeIter>,
    char: char,
}

impl UnicodeLiteralAppender {
    fn new(pred: PrefixRangeIter, char: char) -> Self {
        Self {
            predecessor: Box::new(pred),
            char,
        }
    }
}

impl Iterator for UnicodeLiteralAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
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

impl ExactSizeIterator for UnicodeLiteralAppender {
    fn len(&self) -> usize {
        self.predecessor.len()
    }
}

impl PrefixRangeIterExt for UnicodeLiteralAppender {
    fn prefix_len_bounds(&self) -> (usize, usize) {
        let char_size = self.char.len_utf8();
        let (low, high) = self.predecessor.prefix_len_bounds();
        (low + char_size, high + char_size)
    }
}

#[derive(Clone)]
pub struct ByteClassAppender {
    product: Box<Product<PrefixRangeIter, std::vec::IntoIter<hir::ClassBytesRange>>>,
    len: usize,
    prefix_len_bounds: (usize, usize),
}

impl ByteClassAppender {
    pub fn new(predecessor: PrefixRangeIter, class: hir::ClassBytes) -> Self {
        let (depth_low, depth_high) = predecessor.prefix_len_bounds();
        Self {
            // We shouldn't ever overflow, but panic if we do.
            len: predecessor.len().checked_mul(class.ranges().len()).unwrap(),
            product: Box::new(predecessor.cartesian_product(class.ranges().to_vec())),
            prefix_len_bounds: (depth_low + 1, depth_high + 1),
        }
    }
}

impl Iterator for ByteClassAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        let (curr, range) = self.product.next()?;
        let (mut start, mut end) = curr.into_inner();
        start.push(range.start());
        end.push(range.end());
        Some(start..=end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Note:
        self.product.size_hint()
    }
}

impl ExactSizeIterator for ByteClassAppender {
    fn len(&self) -> usize {
        // Note: we can't use self.product.len() because product is not an ExactSizeIterator
        // because in the general case, the product of two usize could overflow usize. We should
        // never get that large, so we manually calculate len during construction.
        self.len
    }
}

impl PrefixRangeIterExt for ByteClassAppender {
    fn prefix_len_bounds(&self) -> (usize, usize) {
        self.prefix_len_bounds
    }
}

#[derive(Clone)]
pub struct UnicodeClassAppender {
    product: Box<Product<PrefixRangeIter, std::vec::IntoIter<hir::ClassUnicodeRange>>>,
    prefix_len_bounds: (usize, usize),
    len: usize,
}

impl UnicodeClassAppender {
    pub fn new(predecessor: PrefixRangeIter, class: hir::ClassUnicode) -> Self {
        let (depth_low, depth_high) = predecessor.prefix_len_bounds();
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
            // We shouldn't ever overflow, but panic if we do.
            len: predecessor.len().checked_mul(class.ranges().len()).unwrap(),
            product: Box::new(
                predecessor.cartesian_product(split_unicode_ranges(class).into_iter()),
            ),
            prefix_len_bounds: (depth_low + min_char_len, depth_high + max_char_len),
        }
    }
}

impl Iterator for UnicodeClassAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        let (lit_range, unicode_range) = self.product.next()?;
        let (mut start, mut end) = lit_range.into_inner();
        let mut buf = [0u8; 4];
        start.extend_from_slice(unicode_range.start().encode_utf8(&mut buf).as_bytes());
        end.extend_from_slice(unicode_range.end().encode_utf8(&mut buf).as_bytes());
        Some(start..=end)
    }
}

impl ExactSizeIterator for UnicodeClassAppender {
    fn len(&self) -> usize {
        // Note: we can't use self.product.len() because product is not an ExactSizeIterator
        // because in the general case, the product of two usize could overflow usize. We should
        // never get that large, so we manually calculate len during construction.
        self.len
    }
}

impl PrefixRangeIterExt for UnicodeClassAppender {
    fn prefix_len_bounds(&self) -> (usize, usize) {
        self.prefix_len_bounds
    }
}

#[derive(Clone)]
pub struct AlternationAppender {
    product: Box<
        itertools::Product<
            PrefixRangeIter,
            std::iter::Flatten<std::vec::IntoIter<PrefixRangeIter>>,
        >,
    >,
    prefix_len_bounds: (usize, usize),
    len: usize,
}

impl AlternationAppender {
    fn new(pred: PrefixRangeIter, alts: Vec<PrefixRangeIter>) -> Self {
        assert!(!alts.is_empty());
        let (base_low, base_high) = pred.prefix_len_bounds();
        let (alt_low, alt_high) = alts.iter().fold((usize::MAX, 0), |acc, it| {
            let (l, h) = it.prefix_len_bounds();
            let new_min = usize::min(acc.0, l);
            let new_max = usize::max(acc.1, h);
            (new_min, new_max)
        });
        Self {
            len: pred.len().checked_mul(alts.len()).unwrap(),
            product: Box::new(pred.cartesian_product(alts.into_iter().flatten())),
            prefix_len_bounds: (base_low + alt_low, base_high + alt_high),
            // We shouldn't ever overflow, but panic if we do.
        }
    }
}

impl Iterator for AlternationAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<Self::Item> {
        let (base, addition) = self.product.next()?;
        let (mut start, mut end) = base.into_inner();
        start.extend_from_slice(addition.start());
        end.extend_from_slice(addition.end());
        Some(start..=end)
    }
}

impl ExactSizeIterator for AlternationAppender {
    fn len(&self) -> usize {
        // Note: we can't use self.product.len() because product is not an ExactSizeIterator
        // because in the general case, the product of two usize could overflow usize. We should
        // never get that large, so we manually calculate len during construction.
        self.len
    }
}

impl PrefixRangeIterExt for AlternationAppender {
    fn prefix_len_bounds(&self) -> (usize, usize) {
        self.prefix_len_bounds
    }
}

// splits a unicode class into a set of ranges that each have a uniform length when
// encoded to utf8. This allows us to generate accurate prefixes and depth hints
// for unicode.
fn split_unicode_ranges(class: hir::ClassUnicode) -> Vec<hir::ClassUnicodeRange> {
    fn char_unchecked(c: u32) -> char {
        // TODO use char::from_u32_unchecked once #89259 stabilizes
        char::from_u32(c).unwrap()
    }
    let one_byte_min_char: char = char_unchecked(0x0000);
    let one_byte_max_char: char = char_unchecked(0x007F);
    let two_byte_min_char: char = char_unchecked(0x0080);
    let two_byte_max_char: char = char_unchecked(0x0007FF);
    let three_byte_min_char: char = char_unchecked(0x0800);
    let three_byte_max_char: char = char_unchecked(0xFFFF);
    let four_byte_min_char: char = char_unchecked(0x10000);
    let four_byte_max_char: char = char_unchecked(0x10FFFF);

    let new_range = |a: char, b: char| hir::ClassUnicodeRange::new(a, b);
    let sized_ranges = [
        new_range(one_byte_min_char, one_byte_max_char),
        new_range(two_byte_min_char, two_byte_max_char),
        new_range(three_byte_min_char, three_byte_max_char),
        new_range(four_byte_min_char, four_byte_max_char),
    ];

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

    class
        .ranges()
        .to_vec()
        .into_iter()
        .cartesian_product(sized_ranges.into_iter())
        .filter_map(|(left, right)| intersect(left, right))
        .collect()
}
