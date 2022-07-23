use std::ops::{Index, Range, RangeInclusive};

pub type SuffixIdx = u32;
pub type ContentIdx = u32;

#[derive(Copy, Clone)]
pub struct SuffixArray<'s> {
    array: &'s [ContentIdx],
    content: &'s [u8],
    char_offsets: &'s [SuffixIdx; 256],
}

impl<'s> SuffixArray<'s> {
    pub fn new(
        array: &'s [ContentIdx],
        content: &'s [u8],
        char_offsets: &'s [SuffixIdx; 256],
    ) -> Self {
        Self {
            array,
            content,
            char_offsets,
        }
    }

    pub fn find<T>(&self, range: RangeInclusive<T>) -> Range<SuffixIdx>
    where
        T: AsRef<[u8]>,
    {
        self.find_start(range.start().as_ref())..self.find_end(range.end().as_ref())
    }

    // finds the index of the first suffix whose prefix is greater than or equal to needle
    pub fn find_start(&self, needle: &[u8]) -> SuffixIdx {
        if std::intrinsics::unlikely(needle.len() == 0) {
            return 0;
        }
        let (scope_start, scope) = self.scoped_array(needle[0]);
        scope_start
            + scope.partition_point(|&suf_start| {
                let suf_start = suf_start as usize;
                let suf_end = usize::min(suf_start + needle.len(), self.content.len());
                let suf = &self.content[suf_start..suf_end];
                suf < needle
            }) as SuffixIdx
    }

    // finds the index of the first suffix whose prefix is greater than needle
    pub fn find_end(&self, needle: &[u8]) -> SuffixIdx {
        if std::intrinsics::unlikely(needle.len() == 0) {
            return self.array.len() as SuffixIdx;
        }

        let (scope_start, scope) = self.scoped_array(needle[0]);
        scope_start
            + scope.partition_point(|&suf_start| {
                let suf_start = suf_start as usize;
                let suf_end = usize::min(suf_start + needle.len(), self.content.len());
                let suf = &self.content[suf_start..suf_end];
                suf <= needle
            }) as SuffixIdx
    }

    pub fn scoped_array(&self, scope: u8) -> (u32, &'_ [ContentIdx]) {
        let start = self.char_offsets[scope as usize] as usize;
        let end = if std::intrinsics::likely(scope < u8::MAX) {
            self.char_offsets[scope as usize + 1] as usize
        } else {
            self.array.len()
        };
        (start as u32, &self.array[start..end])
    }
}

impl<'s, T> Index<T> for SuffixArray<'s>
where
    T: SuffixArrayIndex,
{
    type Output = T::Output;

    fn index(&self, idx: T) -> &Self::Output {
        idx.index(self)
    }
}

pub trait SuffixArrayIndex {
    type Output: ?Sized;

    fn index<'a>(&self, sa: &SuffixArray<'a>) -> &'a Self::Output;
}

impl SuffixArrayIndex for Range<SuffixIdx> {
    type Output = [SuffixIdx];

    fn index<'a>(&self, sa: &SuffixArray<'a>) -> &'a Self::Output {
        &sa.array[self.start as usize..self.end as usize]
    }
}

impl SuffixArrayIndex for SuffixIdx {
    type Output = SuffixIdx;

    fn index<'a>(&self, sa: &SuffixArray<'a>) -> &'a Self::Output {
        &sa.array[*self as usize]
    }
}
