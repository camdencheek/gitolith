use std::{ops::Range, sync::Arc};

use crate::shard::suffix::CompressedTrigramPointers;

use super::regex::{ConcatLiteralSet, ExtractedRegexLiterals, LiteralSet};

// pub enum OptimizedLiterals {
//     OrderedExact(Vec<ConcatLiteralSet>),
//     Inexact(Vec<OptimizedLiterals>),
//     None,
// }

// pub fn optimize_extracted(
//     extracted: ExtractedRegexLiterals,
//     trigrams: &Arc<CompressedTrigramPointers>,
// ) -> OptimizedLiterals {
//     use ExtractedRegexLiterals::*;

//     match extracted {
//         // We don't handle exact matches specially right now,
//         // so just optimize it as inexact.
//         // Exact(set) => optimize_inexact_literals(vec![set], trigrams),
//         // TODO make sure exact doesn't get too long.
//         Exact(set) => Exact(set),
//         Inexact(sets) => optimize_inexact_literals(sets),
//         None => None,
//     }
// }

// fn optimize_exact_literals(concat: ConcatLiteralSet) -> ExtractedRegexLiterals {
//     let cardinality_limit = 1024; // TODO tune this parameter

//     if concat.cardinality() <= cardinality_limit {
//         return ExtractedRegexLiterals::Exact(concat);
//     }
// }

// fn optimize_inexact_literals(sets: Vec<ConcatLiteralSet>) -> ExtractedRegexLiterals {
//     let mut sets: Vec<ConcatLiteralSet> = sets
//         .into_iter()
//         .map(|set| optimize_prefix_range_set(set))
//         .flatten()
//         .collect();

//     if sets.len() == 0 {
//         return ExtractedRegexLiterals::None;
//     }

//     sets.truncate(3);
//     ExtractedRegexLiterals::Inexact(sets)
// }

// fn optimize_prefix_range_set(set: ConcatLiteralSet) -> Vec<ConcatLiteralSet> {
//     let max_len = 256;
//     let total_len = set.cardinality();
//     if total_len < max_len {
//         return vec![set];
//     }

//     let lits = set.sets();
//     let mut res = Vec::new();
//     for num_slices in 2..lits.len() {
//         res.clear();
//         let target_product = (total_len as f64).powf(1f64 / num_slices as f64) as usize;
//         if target_product > max_len {
//             continue;
//         }

//         let mut start = 0;
//         for slice_num in 0..num_slices - 1 {
//             for end in start + 1..lits.len() {
//                 let slice_len: usize = lits[start..end]
//                     .iter()
//                     .map(LiteralSet::cardinality)
//                     .product();
//                 if slice_len > target_product {
//                     res.push(lits[start..end - 1].to_vec());
//                     start = end - 1;
//                     break;
//                 }
//             }
//         }
//         let last_slice = &lits[start..];
//         if last_slice
//             .iter()
//             .map(LiteralSet::cardinality)
//             .product::<usize>()
//             > max_len
//         {
//             continue;
//         }
//         res.push(last_slice.to_vec());
//         return res.into_iter().map(ConcatLiteralSet::new).collect();
//     }

//     // I'm not 100% convinced the above loop will always terminate, so as
//     // a safety measure, always just return the original if optimization fails.
//     vec![set]
// }

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
