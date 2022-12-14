use itertools::Itertools;
use regex_syntax::hir::{self, Hir};

#[derive(Clone, Debug)]
pub enum ExtractedRegexLiterals {
    /// An exact extraction indicates that every prefix yielded by the iterator is an exact match to
    /// the input regex pattern and does not need to be rechecked with the original regex pattern as
    /// long as it doesn't cross any document boundaries.
    Exact(ConcatLiteralSet),

    /// An inexact extraction requires any matched document to be rechecked. It guarantees that the
    /// only documents that can possibly match the original regex query will contain at least one of
    /// the prefixes from each of the iterators.
    ///
    /// As an example, the regex query /ab(c|d).*ef(g|h)/ will yield something like
    /// Inexact(vec![[abc..abd], [efg..efh]])
    Inexact(Vec<ConcatLiteralSet>),

    /// If no meaningful literals can be extracted from the regex pattern, like /.*/,
    /// the result of extraction will be None.
    None,
}

impl ExtractedRegexLiterals {
    pub fn to_lower_ascii(&self) -> Self {
        match self {
            Self::Exact(cls) => {
                let (cls, exact) = cls.to_lower_ascii();
                if exact {
                    Self::Exact(cls)
                } else {
                    Self::Inexact(vec![cls])
                }
            }
            Self::Inexact(clss) => {
                Self::Inexact(clss.iter().map(|cls| cls.to_lower_ascii().0).collect())
            }
            Self::None => Self::None,
        }
    }
}

/// A concatenation of LiteralSets.
#[derive(Clone, Debug)]
pub struct ConcatLiteralSet(Vec<LiteralSet>);

impl ConcatLiteralSet {
    pub fn new(concat: Vec<LiteralSet>) -> Self {
        Self(concat)
    }

    /// The number of literals represented by this ConcatLiteralSet.
    /// The n-th literal range can be extracted using `write_state_to(i, ...)`.
    pub fn cardinality(&self) -> usize {
        self.0.iter().map(LiteralSet::cardinality).product()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn to_lower_ascii(&self) -> (Self, bool) {
        let mut all_exact = true;
        let new = self
            .0
            .iter()
            .map(|ls| {
                let (ls, exact) = ls.to_ascii_lower();
                all_exact &= exact;
                ls
            })
            .collect();
        (Self(new), all_exact)
    }

    pub fn as_slice(&self) -> &[LiteralSet] {
        self.0.as_ref()
    }

    pub fn write_nth_literal_to(&self, mut state: usize, dst: &mut Vec<u8>) {
        debug_assert!(state < self.cardinality());

        // TODO document this. Intuition is treating the PrefixRangeSet as a mixed-radix number
        // where digit[i] has base self.0[i].len(). We do this because we don't want to allocate
        // extra and then we can iterate over the states with a single, incrementing usize.
        let mut place_value = self.cardinality();
        for set in self.0.iter() {
            place_value /= set.cardinality();
            let digit = state / place_value;
            state %= place_value;
            set.write_state_to(digit, dst);
        }
    }
}

#[derive(Clone, Debug)]
pub enum LiteralSet {
    Byte(u8),
    Unicode(char),
    ByteClass(Vec<hir::ClassBytesRange>),
    UnicodeClass(Vec<hir::ClassUnicodeRange>),
    Alternation(Vec<ConcatLiteralSet>),
}

impl LiteralSet {
    pub fn cardinality(&self) -> usize {
        use LiteralSet::*;

        match self {
            Byte(_) => 1,
            Unicode(_) => 1,
            ByteClass(v) => v
                .iter()
                .map(|range| range.end() as usize - range.start() as usize + 1)
                .sum(),
            UnicodeClass(v) => v
                .iter()
                .map(|range| (range.end() as u32 - range.start() as u32) as usize + 1)
                .sum(),
            Alternation(v) => v.iter().map(|s| s.cardinality()).sum(),
        }
    }

    pub fn to_ascii_lower(&self) -> (Self, bool) {
        use hir::{ClassBytes, ClassBytesRange, ClassUnicode, ClassUnicodeRange};
        use LiteralSet::*;

        match self {
            Byte(b @ b'A'..=b'Z') => (Byte(b.to_ascii_lowercase()), false),
            Byte(b @ b'a'..=b'z') => (Byte(*b), false),
            Byte(b) => (Byte(*b), true),

            Unicode(c @ 'A'..='Z') => (Unicode(c.to_ascii_lowercase()), false),
            Unicode(c @ 'a'..='z') => (Unicode(*c), false),
            Unicode(c) => (Unicode(*c), true),

            ByteClass(bc) => {
                let mut class = ClassBytes::new(bc.clone());

                let uppers = ClassBytes::new([ClassBytesRange::new(b'A', b'Z')]);
                let lowers = ClassBytes::new([ClassBytesRange::new(b'a', b'z')]);

                let uppers_in_class = {
                    let mut u = uppers;
                    u.intersect(&class);
                    u
                };

                let lowers_in_class = {
                    let mut l = lowers;
                    l.intersect(&class);
                    l
                };

                let uppers_in_class_as_lowers =
                    ClassBytes::new(uppers_in_class.clone().iter().map(|cls| {
                        hir::ClassBytesRange::new(
                            cls.start().to_ascii_lowercase(),
                            cls.end().to_ascii_lowercase(),
                        )
                    }));

                let uppers_and_lowers_are_equivalent = {
                    let mut l = lowers_in_class;
                    l.symmetric_difference(&uppers_in_class_as_lowers);
                    l.ranges().is_empty()
                };

                class.difference(&uppers_in_class);
                class.union(&uppers_in_class_as_lowers);
                (
                    ByteClass(class.ranges().to_vec()),
                    uppers_and_lowers_are_equivalent,
                )
            }
            UnicodeClass(uc) => {
                let mut class = ClassUnicode::new(uc.clone());

                let uppers = ClassUnicode::new([ClassUnicodeRange::new('A', 'Z')]);
                let lowers = ClassUnicode::new([ClassUnicodeRange::new('a', 'z')]);

                let uppers_in_class = {
                    let mut u = uppers;
                    u.intersect(&class);
                    u
                };

                let lowers_in_class = {
                    let mut l = lowers;
                    l.intersect(&class);
                    l
                };

                let uppers_in_class_as_lowers =
                    ClassUnicode::new(uppers_in_class.clone().iter().map(|cls| {
                        hir::ClassUnicodeRange::new(
                            cls.start().to_ascii_lowercase(),
                            cls.end().to_ascii_lowercase(),
                        )
                    }));

                let uppers_and_lowers_are_equivalent = {
                    let mut l = lowers_in_class;
                    l.symmetric_difference(&uppers_in_class_as_lowers);
                    l.ranges().is_empty()
                };

                class.difference(&uppers_in_class);
                class.union(&uppers_in_class_as_lowers);
                (
                    UnicodeClass(class.ranges().to_vec()),
                    uppers_and_lowers_are_equivalent,
                )
            }
            Alternation(v) => {
                let mut all_exact = true;
                let v = v
                    .iter()
                    .map(|cls| {
                        let (cls, exact) = cls.to_lower_ascii();
                        all_exact &= exact;
                        cls
                    })
                    .collect();
                (Alternation(v), all_exact)
            }
        }
    }

    fn write_state_to(&self, mut state: usize, dst: &mut Vec<u8>) {
        use LiteralSet::*;

        match self {
            Byte(b) => {
                debug_assert!(state == 0);
                dst.push(*b);
            }
            Unicode(c) => {
                debug_assert!(state == 0);
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                dst.extend_from_slice(bytes);
            }
            ByteClass(v) => {
                fn class_len(class: &hir::ClassBytesRange) -> usize {
                    // class.end() is inclusive, so add 1
                    (class.end() - class.start()) as usize + 1
                }
                for class in v {
                    if state >= class_len(class) {
                        state -= class_len(class);
                        continue;
                    }
                    let b = class.start() + state as u8;
                    dst.push(b);
                    break;
                }
            }
            UnicodeClass(v) => {
                fn class_len(class: &hir::ClassUnicodeRange) -> usize {
                    // class.end() is inclusive, so add 1
                    (class.end() as u32 - class.start() as u32) as usize + 1
                }
                for class in v {
                    if state >= class_len(class) {
                        state -= class_len(class);
                        continue;
                    }
                    let c = char::from_u32(class.start() as u32 + state as u32).unwrap();
                    let mut buf = [0u8; 4];
                    let bytes = c.encode_utf8(&mut buf).as_bytes();
                    dst.extend_from_slice(bytes);
                    break;
                }
            }
            Alternation(v) => {
                let mut state = state;
                for concat in v {
                    if state >= concat.cardinality() {
                        state -= concat.cardinality();
                        continue;
                    }
                    concat.write_nth_literal_to(state, dst);
                    break;
                }
            }
        };
    }
}

pub fn extract_regex_literals(hir: Hir) -> ExtractedRegexLiterals {
    struct Env {
        current: Vec<LiteralSet>,
        complete: Vec<ConcatLiteralSet>,
        exact: bool,
    }

    impl Env {
        fn cannot_handle(&mut self) {
            self.close_current();
            self.exact = false;
        }

        fn close_current(&mut self) {
            if !self.current.is_empty() {
                self.complete
                    .push(ConcatLiteralSet::new(std::mem::take(&mut self.current)));
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
            // TODO we can get some useful information from "one or more" repetitions
            // because we can add the first guaranteed char.
            _ => env.cannot_handle(),
        }
    }

    let mut env = Env {
        current: Vec::new(),
        complete: Vec::new(),
        exact: true,
    };

    push_hir(&mut env, hir);

    match (env.current.len(), env.exact) {
        (0, true) => ExtractedRegexLiterals::None,
        (_, true) => {
            assert!(env.complete.is_empty());
            ExtractedRegexLiterals::Exact(ConcatLiteralSet::new(env.current))
        }
        (_, false) => {
            if !env.current.is_empty() {
                env.complete.push(ConcatLiteralSet::new(env.current));
            }
            if !env.complete.is_empty() {
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
// TODO this function can probably be simplified
fn split_unicode_ranges(ranges: &[hir::ClassUnicodeRange]) -> Vec<hir::ClassUnicodeRange> {
    fn char_unchecked(c: u32) -> char {
        // NOTE: we can use char::from_u32_unchecked once #89259 stabilizes
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

    let sized_ranges = [
        hir::ClassUnicodeRange::new(one_byte_min_char, one_byte_max_char),
        hir::ClassUnicodeRange::new(two_byte_min_char, two_byte_max_char),
        hir::ClassUnicodeRange::new(three_byte_min_char, three_byte_max_char),
        hir::ClassUnicodeRange::new(four_byte_min_char, four_byte_max_char),
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
        .cloned()
        .cartesian_product(sized_ranges.into_iter())
        .filter_map(|(left, right)| intersect(left, right))
        .collect()
}
