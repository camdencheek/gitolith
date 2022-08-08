use std::{ops::Range, sync::Arc};

use crate::shard::suffix::CompressedTrigramPointers;

use super::regex::{ConcatLiteralSet, ExtractedRegexLiterals, LiteralSet};

#[derive(Debug)]
pub enum OptimizedLiterals {
    OrderedExact(Vec<ConcatLiteralSet>),
    Inexact(Vec<ConcatLiteralSet>),
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
    let cardinality_limit = 2048; // TODO tune this parameter

    for n in 1..=concat.as_ref().len() {
        let split = {
            let base_len = concat.as_ref().len() / n;
            let num_with_bonus = concat.as_ref().len() % n;
            let mut v = Vec::with_capacity(n);
            let mut start = 0;
            for i in 0..n {
                let len = if i < num_with_bonus {
                    base_len + 1
                } else {
                    base_len
                };
                v.push(ConcatLiteralSet::new(
                    concat.as_ref()[start..start + len].to_vec(),
                ));
                start = start + len;
            }
            v
        };

        let split_cardinality = split
            .iter()
            .map(ConcatLiteralSet::cardinality)
            .sum::<usize>();

        if split_cardinality < cardinality_limit * n {
            return OptimizedLiterals::OrderedExact(split);
        }
    }

    return optimize_inexact_literals(vec![concat]);
}

fn optimize_inexact_literals(sets: Vec<ConcatLiteralSet>) -> OptimizedLiterals {
    todo!()
}
