use super::docs::{Doc, DocID, DocSlice};
use super::suffix::{SuffixArray, SuffixIdx};
use super::Shard;
use itertools::{Itertools, Product};
use rayon::prelude::*;
use regex::bytes::Regex;
use regex_syntax::hir::{self, Hir};
use roaring::bitmap::RoaringBitmap;
use std::ops::{Range, RangeInclusive};

pub struct DocMatches<'a> {
    pub doc: Doc<'a>,
    pub matches: Vec<Range<u32>>,
}

struct CheckingDocMatchIterator<'a, 'b, T>
where
    T: Iterator<Item = Doc<'a>>,
{
    docs: T,
    checked_docs: usize,
    re: &'b Regex,
}

impl<'a, 'b, T> CheckingDocMatchIterator<'a, 'b, T>
where
    T: Iterator<Item = Doc<'a>>,
{
    fn new(docs: T, re: &'b Regex) -> Self {
        Self {
            docs,
            checked_docs: 0,
            re,
        }
    }
}

impl<'a, 'b, T> Iterator for CheckingDocMatchIterator<'a, 'b, T>
where
    T: Iterator<Item = Doc<'a>>,
{
    type Item = DocMatches<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(doc) = self.docs.next() {
            self.checked_docs += 1;
            let matches: Vec<Range<u32>> = self
                .re
                .find_iter(doc.content)
                .map(|m| m.start() as u32..m.end() as u32)
                .collect();
            if matches.len() > 0 {
                return Some(DocMatches { doc, matches });
            }
        }
        println!("Rechecked docs: {}", self.checked_docs);
        None
    }
}

struct DocBitmapAllIterator<'a, T>
where
    T: Iterator<Item = SuffixIdx>,
{
    docs: DocSlice<'a>,
    doc_bitmaps: Vec<RoaringBitmap>,
    suffix_iters: Vec<T>,
}

impl<'a, T> DocBitmapAllIterator<'a, T>
where
    T: Iterator<Item = SuffixIdx>,
{
    pub fn new(docs: DocSlice<'a>, suffix_iters: Vec<T>) -> Self {
        Self {
            docs,
            doc_bitmaps: vec![RoaringBitmap::new(); docs.len() as usize],
            suffix_iters,
        }
    }

    fn advance_child_to_doc(&mut self, child_num: usize, target_doc: DocID) -> bool {
        // Check if we've already seen this doc for this child
        if self.doc_bitmaps[child_num].contains(target_doc) {
            return true;
        }
        // Otherwise, iterate over suffixes until we find one that is contained
        // in the given doc.
        let child = &mut self.suffix_iters[child_num];
        for suffix in child {
            if let Some(doc) = self.docs.find_by_suffix(suffix) {
                // TODO this is probably unnecessary for the current doc,
                // but it's easier to reason about.
                self.doc_bitmaps[child_num].insert(doc.id);

                if doc.id == target_doc {
                    return true;
                }
            }
        }
        false
    }
}

impl<'a, T> Iterator for DocBitmapAllIterator<'a, T>
where
    T: Iterator<Item = SuffixIdx>,
{
    type Item = Doc<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        'outer: loop {
            let next_doc = self.docs.peek()?;

            // If we have already seen this doc from all children, just return it.
            if self.doc_bitmaps.iter().all(|bm| bm.contains(next_doc.id)) {
                return Some(self.docs.next()?);
            }

            // TODO sort the child iterators so we hit the shortest ones first.

            'inner: for child_num in 0..self.suffix_iters.len() {
                let found = self.advance_child_to_doc(child_num, next_doc.id);
                if found {
                    continue 'inner;
                } else {
                    self.docs.next()?;
                    continue 'outer;
                }
            }

            // The current doc was found in all children
            return Some(self.docs.next()?);
        }
    }
}

struct SuffixSortingIterator {
    collected: <Vec<SuffixIdx> as IntoIterator>::IntoIter,
}

impl SuffixSortingIterator {
    fn new<'a>(shard: &'a Shard, prefix_ranges: PrefixRangeIter) -> Self {
        let sa = shard.sa();
        let mut collected = prefix_ranges
            .map(|pr| sa.find(pr))
            .flatten()
            .map(|idx| idx)
            .collect::<Vec<SuffixIdx>>();
        // TODO the collected set of suffixes could be pretty large.
        // We should probably be allocating this with a request-scoped
        // allocator, or a mmap-ed region.
        // TODO we don't need to sort the whole thing up front,
        // we only need to ensure that the next suffix we yield
        // is the minimum of the remaining. We could do some sort
        // of streaming merge sort here to avoid doing extra work.
        collected.par_sort();
        Self {
            collected: collected.into_iter(),
        }
    }
}

impl Iterator for SuffixSortingIterator {
    type Item = SuffixIdx; // TODO type this as SuffixIdx or something

    fn next(&mut self) -> Option<SuffixIdx> {
        self.collected.next()
    }
}

pub struct InexactMergingIterator<'a, T>
where
    T: Iterator<Item = SuffixIdx>,
{
    docs: std::iter::Peekable<DocSlice<'a>>,
    suffix_iterators: Vec<std::iter::Peekable<T>>,
}

impl<'a, T> InexactMergingIterator<'a, T>
where
    T: Iterator<Item = SuffixIdx>,
{
    fn new(docs: DocSlice<'a>, suffixes: Vec<T>) -> Self {
        Self {
            docs: docs.into_iter().peekable(),
            suffix_iterators: suffixes.into_iter().map(|it| it.peekable()).collect(),
        }
    }

    fn advance_docs_to_contain(&mut self, offset: SuffixIdx) -> Option<Doc<'a>> {
        loop {
            let peeked = *self.docs.peek()?;
            if peeked.end() >= offset {
                return Some(peeked);
            }
            self.docs.next();
        }
    }

    // Advances the children until the next yielded suffix is after the provided offset.
    // Returns the min offset and max offset if all children can be advanced to this state,
    // otherwise returns None.
    fn advance_children_to(&mut self, offset: SuffixIdx) -> Option<(SuffixIdx, SuffixIdx)> {
        let (mut min, mut max) = (u32::MAX, 0u32);
        for child in self.suffix_iterators.iter_mut() {
            loop {
                let peeked = *child.peek()?;
                if peeked >= offset {
                    min = min.min(peeked);
                    max = max.max(peeked);
                    break;
                }
                child.next();
            }
        }
        Some((min, max))
    }
}

impl<'a, T> Iterator for InexactMergingIterator<'a, T>
where
    T: Iterator<Item = SuffixIdx>,
{
    type Item = Doc<'a>;

    fn next(&mut self) -> Option<Doc<'a>> {
        let mut next_doc = *self.docs.peek()?;
        let (mut child_min, mut child_max) = self.advance_children_to(next_doc.start())?;
        loop {
            assert!(child_min >= next_doc.start());
            if next_doc.end() >= child_max {
                return Some(self.docs.next()?);
            }
            next_doc = self.advance_docs_to_contain(child_max)?;
            assert!(next_doc.end() >= child_max);
            if next_doc.start() <= child_min {
                return Some(self.docs.next()?);
            }
            (child_min, child_max) = self.advance_children_to(next_doc.start())?;
        }
    }
}

pub fn new_regex_iter<'a, 'b: 'a>(
    shard: &'a Shard,
    re: &'b Regex,
) -> Box<dyn Iterator<Item = DocMatches<'a>> + 'a> {
    let ast = regex_syntax::ast::parse::Parser::new()
        .parse(re.as_str())
        .expect("regex str failed to parse as AST");
    let hir = regex_syntax::hir::translate::Translator::new()
        .translate(re.as_str(), &ast)
        .expect("regex str failed to parse for translator");
    let (ranges, _exact) = RegexRangesBuilder::from_hir(hir);

    if ranges.len() == 0 {
        // The suffix array provides us no useful information, so just search all the docs.
        Box::new(CheckingDocMatchIterator::new(shard.docs().into_iter(), re))
    // TODO implement an exact iterator
    // } else if self.exact {
    //     assert!(self.complete.len() == 1);
    //     ExactDocIterator::new(self.complete[0])
    // } else {
    //     Box::new(CheckingDocMatchIterator::new(
    //         InexactMergingIterator::new(
    //             shard.docs(),
    //             ranges
    //                 .into_iter()
    //                 .map(|it| SuffixSortingIterator::new(shard, it))
    //                 .collect(),
    //         ),
    //         re,
    //     ))
    } else {
        let sa = shard.sa();
        let prs = ranges
            .into_iter()
            .map(|prefix_iter| {
                prefix_iter
                    .map(move |prefix_range| sa.find(prefix_range))
                    .flatten()
            })
            .collect();
        Box::new(CheckingDocMatchIterator::new(
            DocBitmapAllIterator::new(shard.docs(), prs),
            re,
        ))
    }
}

struct RegexRangesBuilder {
    // An iterator over the current set of prefixes being built, or
    // None if no prefix set has been started.
    current: Option<PrefixRangeIter>,

    // A set of prefix ranges that have already been extended to their
    // maximum length given a pattern.
    complete: Vec<PrefixRangeIter>,

    // True if this builder has not encountered any regex constructs that
    // cannot be handled exactly by the suffix array search and we must
    // rerun the pattern over candidate documents.
    exact: bool,
}

impl RegexRangesBuilder {
    pub fn from_hir(hir: Hir) -> (Vec<PrefixRangeIter>, bool) {
        let mut s = Self {
            current: None,
            complete: Vec::new(),
            exact: true,
        };
        s.push_hir(hir);
        s.build()
    }

    // build finalizes the builder and returns a set of prefix
    // ranges and whether the prefixes
    fn build(mut self) -> (Vec<PrefixRangeIter>, bool) {
        self.close_current();
        (self.complete, self.exact)
    }

    fn push_hir(&mut self, hir: Hir) {
        use regex_syntax::hir::HirKind::*;

        match hir.into_kind() {
            Empty => {}
            Literal(lit) => self.push_literal(lit),
            Class(class) => self.push_class(class),
            Group(g) => self.push_hir(*g.hir),
            Concat(hirs) => self.push_concat(hirs),
            Alternation(hirs) => self.push_alternation(hirs),
            // Everything below this line does not help
            // narrow the search. TODO: see if we can maintain
            // some structure here so we can use these to filter
            // after we've generated suffix ranges.
            Anchor(a) => self.push_anchor(a),
            WordBoundary(wb) => self.push_word_boundary(wb),
            Repetition(rep) => self.push_repetition(rep),
        }
    }

    fn close_current(&mut self) {
        if let Some(open) = self.current.take() {
            if let (_, Some(max)) = open.depth_hint() && max > 3 {
                self.complete.push(open);
            }
            self.current = None;
        }
    }

    // cannot_handle marks the current set of prefixes as complete
    // and marks the document iterator as inexact.
    fn cannot_handle(&mut self) {
        self.close_current();
        self.exact = false;
    }

    fn get_open_mut(&mut self) -> &mut PrefixRangeIter {
        if self.current.is_none() {
            self.current = Some(PrefixRangeIter::Empty(EmptyIterator::new()));
        }

        self.current
            .as_mut()
            .expect("just set the option, should be not be none")
    }

    fn push_concat(&mut self, hirs: Vec<Hir>) {
        for child in hirs.into_iter() {
            self.push_hir(child);
        }
    }

    fn push_literal(&mut self, lit: hir::Literal) {
        use hir::Literal;
        let open = self.get_open_mut();

        // TODO it feels like I shouldn't need the clones here.
        *open = match lit {
            Literal::Byte(b) => {
                PrefixRangeIter::ByteLiteral(ByteLiteralAppender::new(open.clone(), b))
            }
            Literal::Unicode(char) => {
                PrefixRangeIter::UnicodeLiteral(UnicodeLiteralAppender::new(open.clone(), char))
            }
        }
    }

    fn push_class(&mut self, class: hir::Class) {
        use hir::Class;

        let open = self.get_open_mut();

        // TODO it feels like I shouldn't need the clones here.
        *open = match class {
            Class::Bytes(ref bc) => {
                PrefixRangeIter::ByteClass(ByteClassAppender::new(open.clone(), bc.clone()))
            }
            Class::Unicode(ref uc) => {
                PrefixRangeIter::UnicodeClass(UnicodeClassAppender::new(open.clone(), uc.clone()))
            }
        }
    }

    fn push_alternation(&mut self, alts: Vec<Hir>) {
        // Alternation is tricky. If the alternation is all literals/classes, we can handle this
        // efficiently, but if if there are anchors or repetitions, it becomes much more difficult.
        // We take the easy path and only handle alternations with simple children.

        // A method-scoped type that is just used to visit each node of the Hir
        // and report whether there are any Anchors, WordBoundaries, or Repetitions.
        struct SimpleHirChecker {}

        impl hir::Visitor for SimpleHirChecker {
            type Output = ();
            type Err = ();

            fn finish(self) -> Result<Self::Output, Self::Err> {
                Ok(())
            }

            fn visit_pre(&mut self, hir: &Hir) -> Result<(), Self::Err> {
                use hir::HirKind::*;

                match hir.kind() {
                    Anchor(_) | WordBoundary(_) | Repetition(_) => Err(()),
                    _ => Ok(()),
                }
            }
        }

        for alt in &alts {
            if hir::visit(alt, SimpleHirChecker {}).is_err() {
                self.cannot_handle();
                return;
            };
        }

        // If we got here, all children are simple patterns.
        let mut alt_iters = Vec::with_capacity(alts.len());
        for alt in alts {
            let (mut built, _) = Self::from_hir(alt);
            debug_assert!(built.len() == 1);
            alt_iters.push(built.pop().unwrap());
        }

        let open = self.get_open_mut();
        *open = PrefixRangeIter::Alternation(AlternationAppender::new(open.clone(), alt_iters))
    }

    fn push_anchor(&mut self, _anchor: hir::Anchor) {
        self.cannot_handle();
    }

    fn push_word_boundary(&mut self, _wb: hir::WordBoundary) {
        self.cannot_handle();
    }

    fn push_repetition(&mut self, _rep: hir::Repetition) {
        self.cannot_handle();
    }
}

// A range of byte string literals.
// Invariant: start <= end.
type PrefixRange = RangeInclusive<Vec<u8>>;

trait SuffixRangeIterator: Iterator<Item = PrefixRange> {
    // A lower and optional upper bound on the length of the byte vec bounds
    // on the yielded LitRange.
    fn depth_hint(&self) -> (usize, Option<usize>);

    // An estimate of the "selectivity" of the set of suffix ranges yielded
    // by the iterator. Expressed as a proportion of suffixes that are expected
    // to match given an approximately random text.
    fn selectivity_hint(&self) -> f64;
}

#[derive(Clone)]
pub enum PrefixRangeIter {
    Empty(EmptyIterator),
    ByteLiteral(ByteLiteralAppender),
    UnicodeLiteral(UnicodeLiteralAppender),
    ByteClass(ByteClassAppender),
    UnicodeClass(UnicodeClassAppender),
    Alternation(AlternationAppender),
}

impl Iterator for PrefixRangeIter {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<Self::Item> {
        use PrefixRangeIter::*;
        match self {
            Empty(v) => v.next(),
            ByteLiteral(v) => v.next(),
            UnicodeLiteral(v) => v.next(),
            ByteClass(v) => v.next(),
            UnicodeClass(v) => v.next(),
            Alternation(v) => v.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        use PrefixRangeIter::*;
        match self {
            Empty(v) => v.size_hint(),
            ByteLiteral(v) => v.size_hint(),
            UnicodeLiteral(v) => v.size_hint(),
            ByteClass(v) => v.size_hint(),
            UnicodeClass(v) => v.size_hint(),
            Alternation(v) => v.size_hint(),
        }
    }
}

impl SuffixRangeIterator for PrefixRangeIter {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        use PrefixRangeIter::*;
        match self {
            Empty(v) => v.depth_hint(),
            ByteLiteral(v) => v.depth_hint(),
            UnicodeLiteral(v) => v.depth_hint(),
            ByteClass(v) => v.depth_hint(),
            UnicodeClass(v) => v.depth_hint(),
            Alternation(v) => v.depth_hint(),
        }
    }

    fn selectivity_hint(&self) -> f64 {
        use PrefixRangeIter::*;
        match self {
            Empty(v) => v.selectivity_hint(),
            ByteLiteral(v) => v.selectivity_hint(),
            UnicodeLiteral(v) => v.selectivity_hint(),
            ByteClass(v) => v.selectivity_hint(),
            UnicodeClass(v) => v.selectivity_hint(),
            Alternation(v) => v.selectivity_hint(),
        }
    }
}

#[derive(Clone)]
pub struct EmptyIterator(std::iter::Once<PrefixRange>);

impl EmptyIterator {
    fn new() -> Self {
        Self(std::iter::once(b"".to_vec()..=b"".to_vec()))
    }
}

impl Iterator for EmptyIterator {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl SuffixRangeIterator for EmptyIterator {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }

    fn selectivity_hint(&self) -> f64 {
        1.0
    }
}

#[derive(Clone)]
pub struct ByteLiteralAppender {
    predecessor: Box<PrefixRangeIter>,
    byte: u8,
}

impl ByteLiteralAppender {
    fn new(predecessor: PrefixRangeIter, byte: u8) -> Self {
        Self {
            predecessor: Box::new(predecessor),
            byte,
        }
    }
}

impl Iterator for ByteLiteralAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        match self.predecessor.next() {
            Some(r) => {
                let (mut start, mut end) = r.into_inner();
                start.push(self.byte);
                end.push(self.byte);
                Some(start..=end)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.predecessor.size_hint()
    }
}

impl SuffixRangeIterator for ByteLiteralAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.predecessor.depth_hint();
        (low + 1, high.map(|i| i + 1))
    }

    fn selectivity_hint(&self) -> f64 {
        // We could use real-word character distributions for this.
        // For now, just estimate that all printable characters
        // are equally distributed.
        self.predecessor.selectivity_hint() / 93.0
    }
}

#[derive(Clone)]
pub struct UnicodeLiteralAppender {
    predecessor: Box<PrefixRangeIter>,
    char: char,
}

impl UnicodeLiteralAppender {
    fn new(pred: PrefixRangeIter, char: char) -> Self {
        Self {
            predecessor: Box::new(pred),
            char,
        }
    }
}

impl Iterator for UnicodeLiteralAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        match self.predecessor.next() {
            Some(r) => {
                let mut buf = [0u8; 4];
                let bytes = self.char.encode_utf8(&mut buf[..]).as_bytes();
                let (mut start, mut end) = r.into_inner();
                start.extend_from_slice(bytes);
                end.extend_from_slice(bytes);
                Some(start..=end)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.predecessor.size_hint()
    }
}

impl SuffixRangeIterator for UnicodeLiteralAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        let char_size = self.char.len_utf8();
        let (low, high) = self.predecessor.depth_hint();
        (low + char_size, high.map(|i| i + char_size))
    }

    fn selectivity_hint(&self) -> f64 {
        if u32::from(self.char) < 128 {
            self.predecessor.selectivity_hint() / 93.0
        } else {
            // Non-ascii unicode literals are more rare
            self.predecessor.selectivity_hint() / 500.0
        }
    }
}

#[derive(Clone)]
pub struct ByteClassAppender {
    product: Box<Product<PrefixRangeIter, std::vec::IntoIter<hir::ClassBytesRange>>>,
    depth_hint: (usize, Option<usize>),
    selectivity: f64,
}

impl ByteClassAppender {
    pub fn new(predecessor: PrefixRangeIter, class: hir::ClassBytes) -> Self {
        let (depth_low, depth_high) = predecessor.depth_hint();
        let selectivity = predecessor.selectivity_hint()
            * class
                .iter()
                .map(|r| (r.end() - r.start()) as f64 / 256f64)
                .sum::<f64>();
        Self {
            product: Box::new(predecessor.cartesian_product(class.ranges().to_vec())),
            depth_hint: (depth_low + 1, depth_high.map(|i| i + 1)),
            selectivity,
        }
    }
}

impl Iterator for ByteClassAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        let (curr, range) = self.product.next()?;
        let (mut start, mut end) = curr.into_inner();
        start.push(range.start());
        end.push(range.end());
        Some(start..=end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.product.size_hint()
    }
}

impl SuffixRangeIterator for ByteClassAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        self.depth_hint
    }

    fn selectivity_hint(&self) -> f64 {
        self.selectivity
    }
}

#[derive(Clone)]
pub struct UnicodeClassAppender {
    product: Box<Product<PrefixRangeIter, UnicodeRangeSplitIterator>>,
    depth_hint: (usize, Option<usize>),
    selectivity: f64,
}

impl UnicodeClassAppender {
    pub fn new(predecessor: PrefixRangeIter, class: hir::ClassUnicode) -> Self {
        let (depth_low, depth_high) = predecessor.depth_hint();
        let min_char_len = class
            .ranges()
            .first()
            .expect("rangess should never be empty")
            .start()
            .len_utf8();
        let max_char_len = class
            .ranges()
            .last()
            .expect("ranges should never be empty")
            .end()
            .len_utf8();

        let selectivity = predecessor.selectivity_hint()
            * class
                .iter()
                .map(|r| (u32::from(r.end()) - u32::from(r.start())) as f64 / 144_697f64)
                .sum::<f64>();
        Self {
            product: Box::new(predecessor.cartesian_product(UnicodeRangeSplitIterator::new(class))),
            depth_hint: (
                depth_low + min_char_len,
                depth_high.map(|i| i + max_char_len),
            ),
            selectivity,
        }
    }
}

impl Iterator for UnicodeClassAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<PrefixRange> {
        let (lit_range, unicode_range) = self.product.next()?;
        let (mut start, mut end) = lit_range.into_inner();
        let mut buf = [0u8; 4];
        start.extend_from_slice(unicode_range.start().encode_utf8(&mut buf).as_bytes());
        end.extend_from_slice(unicode_range.end().encode_utf8(&mut buf).as_bytes());
        Some(start..=end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.product.size_hint()
    }
}

impl SuffixRangeIterator for UnicodeClassAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        self.depth_hint
    }

    fn selectivity_hint(&self) -> f64 {
        self.selectivity
    }
}

#[derive(Clone)]
pub struct AlternationAppender {
    product: Box<
        itertools::Product<
            PrefixRangeIter,
            std::iter::Flatten<std::vec::IntoIter<PrefixRangeIter>>,
        >,
    >,
    depth_hint: (usize, Option<usize>),
    selectivity: f64,
}

impl AlternationAppender {
    fn new(pred: PrefixRangeIter, alts: Vec<PrefixRangeIter>) -> Self {
        assert!(!alts.is_empty());
        let (base_low, base_high) = pred.depth_hint();
        let (alt_low, alt_high) = alts.iter().fold((usize::MAX, Some(0)), |acc, it| {
            let (l, h) = it.depth_hint();
            let new_min = usize::min(acc.0, l);
            let new_max = match (acc.1, h) {
                (Some(i), Some(j)) => Some(usize::max(i, j)),
                _ => None,
            };
            (new_min, new_max)
        });
        let selectivity = pred.selectivity_hint()
            * alts
                .iter()
                .map(PrefixRangeIter::selectivity_hint)
                .sum::<f64>();
        Self {
            product: Box::new(pred.cartesian_product(alts.into_iter().flatten())),
            depth_hint: (
                base_low + alt_low,
                base_high.and_then(|b| alt_high.map(|h| b + h)),
            ),
            selectivity,
        }
    }
}

impl Iterator for AlternationAppender {
    type Item = PrefixRange;

    fn next(&mut self) -> Option<Self::Item> {
        let (base, addition) = self.product.next()?;
        let (mut start, mut end) = base.into_inner();
        start.extend_from_slice(addition.start());
        end.extend_from_slice(addition.end());
        Some(start..=end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.product.size_hint()
    }
}

impl SuffixRangeIterator for AlternationAppender {
    fn depth_hint(&self) -> (usize, Option<usize>) {
        self.depth_hint
    }

    fn selectivity_hint(&self) -> f64 {
        self.selectivity
    }
}

#[derive(Clone)]
struct UnicodeRangeSplitIterator {
    product: Product<
        std::vec::IntoIter<hir::ClassUnicodeRange>,
        std::array::IntoIter<hir::ClassUnicodeRange, 4>,
    >,
}

impl UnicodeRangeSplitIterator {
    fn new(class: hir::ClassUnicode) -> Self {
        let new_range = |a: u32, b: u32| {
            hir::ClassUnicodeRange::new(char::from_u32(a).unwrap(), char::from_u32(b).unwrap())
        };
        // TODO this could probably be const
        let sized_ranges = [
            new_range(0, 0x007F),
            new_range(0x0080, 0x07FF),
            new_range(0x0800, 0xFFFF),
            new_range(0x10000, 0x10FFFF),
        ];
        Self {
            product: class
                .ranges()
                .to_vec()
                .into_iter()
                .cartesian_product(sized_ranges.into_iter()),
        }
    }

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
}

impl Iterator for UnicodeRangeSplitIterator {
    type Item = hir::ClassUnicodeRange;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (left, right) = self.product.next()?;
            if let Some(range) = Self::intersect(left, right) {
                return Some(range);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // TODO: we can get a tighter bound here. We can calculate the size on construction. Given
        // a boundary code point is a code point with utf8 representation of n bytes where the
        // previous codepoint is represented in n-1 bytes, we can just count the number of boundary
        // codepoints contained in the ranges and add it to the number of ranges.
        self.product.size_hint()
    }
}
