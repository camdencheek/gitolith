use std::ops::Range;
use std::{io::Write, ops::RangeInclusive};

use itertools::Itertools;
use regex_syntax::hir;

#[derive(Clone)]
struct PrefixRangeSet(Vec<LiteralSet>);

impl PrefixRangeSet {
    pub fn new(&self, concat: Vec<LiteralSet>) -> Self {
        Self(concat)
    }

    pub fn len(&self) -> usize {
        self.0.iter().map(LiteralSet::len).product()
    }
}

struct PrefixRangeSetIter<'a> {
    set: &'a PrefixRangeSet,
    place_values: Vec<usize>,
    state: Range<usize>,
}

impl<'a> PrefixRangeSetIter<'a> {
    pub fn new(set: &'a PrefixRangeSet) -> Self {
        let mut place = set.len();
        Self {
            set,
            place_values: set
                .0
                .iter()
                .map(|l| {
                    place /= l.len();
                    place
                })
                .collect(),
            state: 0..set.len(),
        }
    }

    pub fn write_state_to(&self, mut state: usize, range: &mut Range<Vec<u8>>) {
        range.start.clear();
        range.end.clear();
        for (i, ls) in self.set.0.iter().enumerate() {
            let rem = state % self.place_values[i];
            let ls_state = state / self.place_values[i];
            self.set.0[i].write_state_to(state, &mut range.start, &mut range.end);
        }
    }
}

impl<'a> Iterator for PrefixRangeSetIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.state.next()
    }
}

#[derive(Clone)]
pub enum LiteralSet {
    Byte(u8),
    Unicode(char),
    ByteClass(Vec<hir::ClassBytesRange>),
    UnicodeClass(Vec<hir::ClassUnicodeRange>),
    Alternation(Vec<LiteralSet>),
}

impl LiteralSet {
    fn len(&self) -> usize {
        use LiteralSet::*;

        match self {
            Byte(_) => 1,
            Unicode(c) => 1,
            ByteClass(v) => v.len(),
            UnicodeClass(v) => v.len(),
            Alternation(v) => v.iter().map(|s| s.len()).sum(),
        }
    }

    fn write_state_to(&self, state: usize, start: &mut Vec<u8>, end: &mut Vec<u8>) {
        use LiteralSet::*;

        match self {
            Byte(b) => {
                debug_assert!(state == 0);
                start.write(&[*b]);
                end.write(&[*b]);
            }
            Unicode(c) => {
                debug_assert!(state == 0);
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                start.write(bytes);
                end.write(bytes);
            }
            ByteClass(v) => {
                let class = v[state];
                start.write(&[class.start()]);
                end.write(&[class.end()]);
            }
            UnicodeClass(v) => {
                let class = v[state];
                let mut buf = [0u8; 4];
                let start_bytes = class.start().encode_utf8(&mut buf).as_bytes();
                start.write(start_bytes);
                let end_bytes = class.start().encode_utf8(&mut buf).as_bytes();
                end.write(end_bytes);
            }
            Alternation(v) => {
                let mut state = state;
                for lc in v {
                    if state >= lc.len() {
                        state -= lc.len();
                        continue;
                    }
                    lc.write_state_to(state, start, end)
                }
            }
        };
    }
}

enum ExtractedRegexLiterals {
    // An exact extraction indicates that every prefix yielded by the iterator is an exact match to
    // the input regex pattern and does not need to be rechecked with the original regex pattern.
    Exact(PrefixRangeSet),

    // An inexact extraction requires any matched document to be rechecked. It guarantees that the
    // only documents that can possibly match the original regex query will contain at least one of
    // the prefixes from each of the iterators.
    //
    // As an example, the regex query /ab(c|d).*ef(g|h)/ will yield something like
    // Inexact(vec![[abc, abd], [efg, efh]])
    Inexact(Vec<PrefixRangeSet>),

    // If no meaningful literals can be extracted from the regex pattern, like /.*/,
    // the result of extraction will be None.
    None,
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

        match (s.current, s.exact) {
            (Some(c), true) => ExtractedRegexLiterals::Exact(c),
            (None, true) => ExtractedRegexLiterals::None,
            (Some(c), false) => {
                s.complete.push(c);
                ExtractedRegexLiterals::Inexact(s.complete)
            }
            (None, false) => {
                if s.complete.len() == 0 {
                    ExtractedRegexLiterals::None
                } else {
                    ExtractedRegexLiterals::Inexact(s.complete)
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

    fn take_current(&mut self) -> PrefixRangeIter {
        match self.current.take() {
            Some(o) => o,
            None => PrefixRangeIter::Empty(EmptyIterator::new()),
        }
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
        let current = self.take_current();

        self.current = match lit {
            Literal::Byte(b) => Some(PrefixRangeIter::ByteLiteral(Box::new(
                ByteLiteralAppender::new(current, b),
            ))),
            Literal::Unicode(char) => Some(PrefixRangeIter::UnicodeLiteral(Box::new(
                UnicodeLiteralAppender::new(current, char),
            ))),
        }
    }

    fn push_class(&mut self, class: hir::Class) {
        use hir::Class;

        let current = self.take_current();

        self.current = match class {
            Class::Bytes(ref bc) => Some(PrefixRangeIter::ByteClass(Box::new(
                ByteClassAppender::new(current, bc.clone()),
            ))),
            Class::Unicode(ref uc) => Some(PrefixRangeIter::UnicodeClass(Box::new(
                UnicodeClassAppender::new(current, uc.clone()),
            ))),
        }
    }

    fn push_alternation(&mut self, alts: Vec<Hir>) {
        let mut alt_iters = Vec::with_capacity(alts.len());
        for alt in alts {
            match Self::from_hir(alt) {
                ExtractedRegexLiterals::Exact(e) => alt_iters.push(e),
                // We don't handle anything but simple alternations right now.
                _ => return self.cannot_handle(),
            }
        }

        let current = self.take_current();
        self.current = Some(PrefixRangeIter::Alternation(Box::new(
            AlternationAppender::new(current, alt_iters),
        )))
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

// splits a unicode class into a set of ranges that each have a uniform length when
// encoded to utf8. This allows us to generate accurate prefixes and depth hints
// for unicode.
fn split_unicode_ranges(ranges: &[hir::ClassUnicodeRange]) -> Vec<hir::ClassUnicodeRange> {
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

    ranges
        .iter()
        .cartesian_product(sized_ranges.into_iter())
        .filter_map(|(left, right)| intersect(*left, right))
        .collect()
}
