pub struct AsciiLowerIter<T> {
    inner: T,
}

impl<T> AsciiLowerIter<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T> Iterator for AsciiLowerIter<T>
where
    T: Iterator<Item = u8>,
{
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.inner.next()?.to_ascii_lowercase())
    }
}

impl<T> DoubleEndedIterator for AsciiLowerIter<T>
where
    T: DoubleEndedIterator<Item = u8>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(self.inner.next_back()?.to_ascii_lowercase())
    }
}

impl<T> ExactSizeIterator for AsciiLowerIter<T>
where
    T: ExactSizeIterator<Item = u8>,
{
    fn len(&self) -> usize {
        self.inner.len()
    }
}
