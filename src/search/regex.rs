use std::ops::Range;
use std::{io::Write, ops::RangeInclusive};

use itertools::Itertools;
use regex_syntax::hir::{self, Hir};

#[derive(Clone)]
pub struct PrefixRangeSet(Vec<LiteralSet>);

impl PrefixRangeSet {
    pub fn new(concat: Vec<LiteralSet>) -> Self {
        Self(concat)
    }

    pub fn len(&self) -> usize {
        self.0.iter().map(LiteralSet::len).product()
    }

    pub fn write_state_to(&self, mut state: usize, start: &mut Vec<u8>, end: &mut Vec<u8>) {
        start.clear();
        end.clear();

        let mut cur_place = self.0.len();
        let with_place_values = self.0.iter().map(|ls| {
            cur_place /= ls.len();
            (ls, cur_place)
        });

        for (ls, place_value) in with_place_values {
            let rem = state % place_value;
            let ls_state = state / place_value;
            ls.write_state_to(state, start, end);
        }
    }
}

#[derive(Clone)]
pub enum LiteralSet {
    Byte(u8),
    Unicode(char),
    ByteClass(Vec<hir::ClassBytesRange>),
    UnicodeClass(Vec<hir::ClassUnicodeRange>),
    Alternation(Vec<PrefixRangeSet>),
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
                for prefix_set in v {
                    if state >= prefix_set.len() {
                        state -= prefix_set.len();
                        continue;
                    }
                    prefix_set.write_state_to(state, start, end)
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
    current: Vec<LiteralSet>,
    complete: Vec<PrefixRangeSet>,
    exact: bool,
}

impl RegexLiteralExtractor {
    pub fn from_hir(hir: Hir) -> ExtractedRegexLiterals {
        let mut s = Self {
            current: Vec::new(),
            complete: Vec::new(),
            exact: true,
        };
        s.push_hir(hir);

        match (s.current.len(), s.exact) {
            (0, true) => ExtractedRegexLiterals::None,
            (_, true) => ExtractedRegexLiterals::Exact(PrefixRangeSet::new(s.current)),
            (_, false) => {
                s.complete.push(PrefixRangeSet::new(s.current));
                ExtractedRegexLiterals::Inexact(s.complete)
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
        self.complete.push(PrefixRangeSet::new(std::mem::replace(
            &mut self.current,
            Vec::new(),
        )));
        self.current = Vec::new();
    }

    // cannot_handle marks the current set of prefixes as complete
    // and marks the document iterator as inexact.
    fn cannot_handle(&mut self) {
        self.close_current();
        self.exact = false;
    }

    fn push_concat(&mut self, hirs: Vec<Hir>) {
        for child in hirs.into_iter() {
            self.push_hir(child);
        }
    }

    fn push_literal(&mut self, lit: hir::Literal) {
        use hir::Literal;

        self.current.push(match lit {
            Literal::Byte(b) => LiteralSet::Byte(b),
            Literal::Unicode(char) => LiteralSet::Unicode(char),
        })
    }

    fn push_class(&mut self, class: hir::Class) {
        use hir::Class;

        self.current.push(match class {
            Class::Bytes(bc) => LiteralSet::ByteClass(bc.ranges().to_vec()),
            Class::Unicode(uc) => LiteralSet::UnicodeClass(uc.ranges().to_vec()),
        })
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

        self.current.push(LiteralSet::Alternation(alt_iters))
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
