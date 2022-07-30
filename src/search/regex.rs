use std::ops::Range;
use std::{io::Write, ops::RangeInclusive};

use itertools::Itertools;
use regex_syntax::hir::{self, Hir};

#[derive(Clone, Debug)]
pub struct PrefixRangeSet(Vec<LiteralSet>);

impl PrefixRangeSet {
    pub fn new(concat: Vec<LiteralSet>) -> Self {
        Self(concat)
    }

    pub fn len(&self) -> usize {
        self.0.iter().map(LiteralSet::len).product()
    }

    pub fn write_state_to(&self, mut state: usize, start: &mut Vec<u8>, end: &mut Vec<u8>) {
        debug_assert!(state < self.len());

        let mut place_value = self.len();
        for ls in self.0.iter() {
            place_value /= ls.len();
            let ls_state = state / place_value;
            state = state % place_value;
            ls.write_state_to(ls_state, start, end);
        }
    }
}

#[derive(Clone, Debug)]
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
                    prefix_set.write_state_to(state, start, end);
                    break;
                }
            }
        };
    }
}

#[derive(Clone, Debug)]
pub enum ExtractedRegexLiterals {
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

pub fn extract_regex_literals(hir: Hir) -> ExtractedRegexLiterals {
    struct Env {
        current: Vec<LiteralSet>,
        complete: Vec<PrefixRangeSet>,
        exact: bool,
    }

    impl Env {
        fn cannot_handle(&mut self) {
            self.close_current();
            self.exact = false;
        }

        fn close_current(&mut self) {
            if self.current.len() > 0 {
                self.complete.push(PrefixRangeSet::new(std::mem::replace(
                    &mut self.current,
                    Vec::new(),
                )));
            }
        }
    }

    fn push_hir(env: &mut Env, hir: Hir) {
        use regex_syntax::hir::{Class, HirKind, Literal};

        match hir.into_kind() {
            HirKind::Literal(lit) => env.current.push(match lit {
                Literal::Byte(b) => LiteralSet::Byte(b),
                Literal::Unicode(char) => LiteralSet::Unicode(char),
            }),
            HirKind::Class(class) => env.current.push(match class {
                Class::Bytes(bc) => LiteralSet::ByteClass(bc.ranges().to_vec()),
                Class::Unicode(uc) => LiteralSet::UnicodeClass(split_unicode_ranges(uc.ranges())),
            }),
            HirKind::Group(g) => push_hir(env, *g.hir),
            HirKind::Concat(hirs) => {
                for child in hirs.into_iter() {
                    push_hir(env, child)
                }
            }
            HirKind::Alternation(hirs) => {
                let mut alt_iters = Vec::with_capacity(hirs.len());
                for alt in hirs {
                    match extract_regex_literals(alt) {
                        ExtractedRegexLiterals::Exact(e) => alt_iters.push(e),
                        _ => {
                            env.cannot_handle();
                            return;
                        }
                    }
                }
                env.current.push(LiteralSet::Alternation(alt_iters))
            }
            _ => env.cannot_handle(),
        }
    };

    let mut env = Env {
        current: Vec::new(),
        complete: Vec::new(),
        exact: true,
    };

    push_hir(&mut env, hir);

    match (env.current.len(), env.exact) {
        (0, true) => ExtractedRegexLiterals::None,
        (_, true) => {
            assert!(env.complete.len() == 0);
            ExtractedRegexLiterals::Exact(PrefixRangeSet::new(env.current))
        }
        (_, false) => {
            if env.current.len() > 0 {
                env.complete.push(PrefixRangeSet::new(env.current));
            }
            if env.complete.len() > 0 {
                ExtractedRegexLiterals::Inexact(env.complete)
            } else {
                ExtractedRegexLiterals::None
            }
        }
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
