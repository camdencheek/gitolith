use crate::shard::docs::DocID;

pub mod regex;

pub struct DocMatch {
    id: DocID,
    matches: Vec<OffsetLen>,
}

pub struct OffsetLen {
    offset: u32,
    len: u32,
}
