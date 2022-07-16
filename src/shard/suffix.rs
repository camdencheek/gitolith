use std::ops::FnOnce;

pub type SuffixIdx = u32;

pub struct SuffixArray<'a>(&'a [u32]);

impl<'a> SuffixArray<'a> {
    pub fn new(sa: &'a [u32]) -> Self {
        Self(sa)
    }
}

impl<'a> AsRef<[u32]> for SuffixArray<'a> {
    fn as_ref(&self) -> &'a [u32] {
        self.0
    }
}
