use super::regex::{ConcatLiteralSet, ExtractedRegexLiterals, LiteralSet};

#[derive(Debug)]
pub enum OptimizedLiterals {
    OrderedExact(Vec<ConcatLiteralSet>),
    Inexact(Vec<ConcatLiteralSet>),
    None,
}

pub fn optimize_extracted(extracted: ExtractedRegexLiterals) -> OptimizedLiterals {
    use ExtractedRegexLiterals::*;

    match extracted {
        Exact(set) => optimize_exact_literals(set),
        Inexact(sets) => optimize_inexact_literals(sets),
        None => OptimizedLiterals::None,
    }
}

fn optimize_exact_literals(concat: ConcatLiteralSet) -> OptimizedLiterals {
    let cardinality_limit = 128; // TODO tune this parameter
    if concat.cardinality() < cardinality_limit {
        return OptimizedLiterals::OrderedExact(vec![concat]);
    }

    optimize_inexact_literals(vec![concat])
}

fn optimize_inexact_literals(concats: Vec<ConcatLiteralSet>) -> OptimizedLiterals {
    let mut res = Vec::with_capacity(concats.len());
    for concat in concats {
        if let Some(mut split) = optimize_inexact_literal(concat) {
            res.append(&mut split);
        } else {
            return OptimizedLiterals::None;
        }
    }
    // Make sure each is a trigram at minimum
    // TODO make this work correctly for alternations. We should maybe
    // just flatten alternations in this representation.
    res.retain(|concat| concat.len() > 2);
    res.sort_by_key(|concat| concat.len());
    res.truncate(4);

    if !res.is_empty() {
        OptimizedLiterals::Inexact(res)
    } else {
        OptimizedLiterals::None
    }
}

fn optimize_inexact_literal(concat: ConcatLiteralSet) -> Option<Vec<ConcatLiteralSet>> {
    let cardinality_limit = 128; // TODO tune this parameter

    for n in 1..=concat.as_slice().len() {
        let split = split_mostly_even(concat.as_slice(), n)
            .iter()
            .map(|lits| ConcatLiteralSet::new(lits.to_vec()))
            .collect::<Vec<_>>();

        let split_cardinality = split
            .iter()
            .map(ConcatLiteralSet::cardinality)
            .sum::<usize>();

        if split_cardinality < cardinality_limit * n {
            return Some(split);
        }
    }

    None
}

fn split_mostly_even<T>(items: &[T], parts: usize) -> Vec<&[T]> {
    let base_len = items.len() / parts;
    let num_with_bonus = items.len() % parts;
    let mut v = Vec::with_capacity(parts);
    let mut start = 0;
    for i in 0..parts {
        let len = if i < num_with_bonus {
            base_len + 1
        } else {
            base_len
        };
        v.push(&items[start..start + len]);
        start += len;
    }
    v
}
