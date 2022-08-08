use std::sync::Arc;

use crate::shard::suffix::CompressedTrigramPointers;

use super::regex::{ConcatLiteralSet, ExtractedRegexLiterals, LiteralSet};

pub fn optimize_extracted(
    extracted: ExtractedRegexLiterals,
    trigrams: &Arc<CompressedTrigramPointers>,
) -> ExtractedRegexLiterals {
    use ExtractedRegexLiterals::*;

    match extracted {
        // We don't handle exact matches specially right now,
        // so just optimize it as inexact.
        // Exact(set) => optimize_inexact_literals(vec![set], trigrams),
        // TODO make sure exact doesn't get too long.
        Exact(set) => Exact(set),
        Inexact(sets) => optimize_inexact_literals(sets, trigrams),
        None => None,
    }
}

fn optimize_inexact_literals(
    sets: Vec<ConcatLiteralSet>,
    trigrams: &Arc<CompressedTrigramPointers>,
) -> ExtractedRegexLiterals {
    let mut sets: Vec<ConcatLiteralSet> = sets
        .into_iter()
        .map(|set| optimize_prefix_range_set(set, &trigrams))
        .flatten()
        .filter(|set| set.selectivity(&trigrams) < 0.0001)
        .collect();

    if sets.len() == 0 {
        return ExtractedRegexLiterals::None;
    }

    sets.sort_by(|a, b| {
        a.selectivity(&trigrams)
            .partial_cmp(&b.selectivity(&trigrams))
            .unwrap()
            .reverse()
    });
    sets.truncate(3);
    ExtractedRegexLiterals::Inexact(sets)
}

fn optimize_prefix_range_set(
    set: ConcatLiteralSet,
    trigrams: &Arc<CompressedTrigramPointers>,
) -> Vec<ConcatLiteralSet> {
    let max_len = 256;
    let total_len = set.cardinality();
    if total_len < max_len {
        return vec![set];
    }

    let lits = set.sets();
    let mut res = Vec::new();
    for num_slices in 2..lits.len() {
        res.clear();
        let target_product = (total_len as f64).powf(1f64 / num_slices as f64) as usize;
        if target_product > max_len {
            continue;
        }

        let mut start = 0;
        for slice_num in 0..num_slices - 1 {
            for end in start + 1..lits.len() {
                let slice_len: usize = lits[start..end]
                    .iter()
                    .map(LiteralSet::cardinality)
                    .product();
                if slice_len > target_product {
                    res.push(lits[start..end - 1].to_vec());
                    start = end - 1;
                    break;
                }
            }
        }
        let last_slice = &lits[start..];
        if last_slice
            .iter()
            .map(LiteralSet::cardinality)
            .product::<usize>()
            > max_len
        {
            continue;
        }
        res.push(last_slice.to_vec());
        return res.into_iter().map(ConcatLiteralSet::new).collect();
    }

    // I'm not 100% convinced the above loop will always terminate, so as
    // a safety measure, always just return the original if optimization fails.
    vec![set]
}
