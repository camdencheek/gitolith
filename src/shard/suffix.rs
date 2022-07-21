use std::ops::FnOnce;

pub type SuffixIdx = u32;

pub type SuffixArray<'a> = &'a [SuffixIdx];
