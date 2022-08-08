use std::{ops::Range, sync::Arc};

use crate::shard::suffix::CompressedTrigramPointers;

use super::regex::{ConcatLiteralSet, ExtractedRegexLiterals, LiteralSet};

pub enum OptimizedLiterals {
    OrderedExact(Vec<ConcatLiteralSet>),
    Inexact(Vec<OptimizedLiterals>),
    None,
}

pub fn optimize_extracted(
    extracted: ExtractedRegexLiterals,
    trigrams: &Arc<CompressedTrigramPointers>,
) -> OptimizedLiterals {
    use ExtractedRegexLiterals::*;

    match extracted {
        Exact(set) => optimize_exact_literals(set),
        Inexact(sets) => optimize_inexact_literals(sets),
        None => OptimizedLiterals::None,
    }
}

fn optimize_exact_literals(concat: ConcatLiteralSet) -> OptimizedLiterals {
    let cardinality_limit = 1024; // TODO tune this parameter

    for n in 1..=concat.as_ref().len() {
        let mut min_cardinality = usize::MAX;
        let mut min_split: Vec<ConcatLiteralSet> = Vec::new();

        // TODO this might be a lot of allocations
        let splits = SplitIterator::new(concat.as_ref(), n).map(|sets| {
            sets.iter()
                .map(|set| ConcatLiteralSet::new(set.to_vec()))
                .collect::<Vec<ConcatLiteralSet>>()
        });

        for split in splits {
            let cardinality = split.iter().map(ConcatLiteralSet::cardinality).sum();
            if cardinality < min_cardinality {
                min_cardinality = cardinality;
                min_split = split;
            }
        }
        if min_cardinality < cardinality_limit * n {
            return OptimizedLiterals::OrderedExact(min_split);
        }
    }

    return optimize_inexact_literals(vec![concat]);
}

fn optimize_inexact_literals(sets: Vec<ConcatLiteralSet>) -> OptimizedLiterals {
    todo!()
}

struct SplitIterator<'a, T> {
    items: &'a [T],
    head: &'a [T],
    tails: Option<Box<SplitIterator<'a, T>>>,
    n: usize,
}

impl<'a, T> SplitIterator<'a, T> {
    fn new(items: &'a [T], n: usize) -> Self {
        Self {
            items,
            head: if n == 1 { &items[0..] } else { &items[0..1] },
            tails: if n == 1 {
                None
            } else {
                Some(Box::new(SplitIterator::new(&items[1..], n - 1)))
            },
            n,
        }
    }
}

impl<'a, T> Iterator for SplitIterator<'a, T> {
    type Item = Vec<&'a [T]>;

    fn next(&mut self) -> Option<Self::Item> {
        let tails = match self.tails.as_mut() {
            None => {
                if self.head.len() > 0 {
                    let h = self.head;
                    self.head = &self.items[..0];
                    return Some(vec![h]);
                } else {
                    return None;
                }
            }
            Some(tails) => tails,
        };

        match tails.next() {
            Some(mut tail) => {
                let mut v = Vec::with_capacity(tail.len() + 1);
                v.push(self.head);
                v.append(&mut tail);
                Some(v)
            }
            None => {
                if self.items.len() - self.head.len() < self.n {
                    None
                } else {
                    self.head = &self.items[0..self.head.len() + 1];
                    let mut tails = Box::new(SplitIterator::new(
                        &self.items[self.head.len()..],
                        self.n - 1,
                    ));
                    let mut tail = tails.next()?;
                    self.tails = Some(tails);
                    let mut v = Vec::with_capacity(tail.len() + 1);
                    v.push(self.head);
                    v.append(&mut tail);
                    Some(v)
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_split_iterator() {
        let two = [1, 2].as_ref();
        let three = [1, 2, 3].as_ref();
        let four = [1, 2, 3, 4].as_ref();

        let cases = [
            (two, 1, vec![vec![&two[0..2]]]),
            (two, 2, vec![vec![&two[0..1], &two[1..2]]]),
            (three, 1, vec![vec![&three[0..3]]]),
            (
                three,
                2,
                vec![
                    vec![&three[0..1], &three[1..3]],
                    vec![&three[0..2], &three[2..3]],
                ],
            ),
            (
                three,
                3,
                vec![vec![&three[0..1], &three[1..2], &three[2..3]]],
            ),
            (
                four,
                2,
                vec![
                    vec![&four[0..1], &four[1..4]],
                    vec![&four[0..2], &four[2..4]],
                    vec![&four[0..3], &four[3..4]],
                ],
            ),
            (
                four,
                3,
                vec![
                    vec![&four[0..1], &four[1..2], &four[2..4]],
                    vec![&four[0..1], &four[1..3], &four[3..4]],
                    vec![&four[0..2], &four[2..3], &four[3..4]],
                ],
            ),
        ];

        for case in cases {
            let got: Vec<Vec<&[i32]>> = SplitIterator::new(case.0, case.1).collect();
            assert_eq!(
                case.2, got,
                "failed for test case {:?}, {:?}",
                case.0, case.1
            )
        }
    }
}
