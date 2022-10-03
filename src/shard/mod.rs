pub mod builder;
pub mod cached;
pub mod cached_file;
pub mod docs;
pub mod file;
pub mod suffix;

use self::file::{ShardFile, ShardStore};
use anyhow::Error;
use std::fs::File;
use std::path::Path;

// use super::cache::Cache;
use super::shard::suffix::SuffixArrayStore;
use derive_more::{Add, From, Into, Sub};
use docs::DocStore;
use std::sync::Arc;

#[derive(Copy, Clone, From, Into, Add, Sub, PartialEq, Eq, Hash, Debug)]
pub struct ShardID(pub u16);

impl From<ShardID> for u64 {
    fn from(shard_id: ShardID) -> Self {
        shard_id.0 as u64
    }
}

#[derive(Clone)]
pub struct Shard {
    pub file: ShardStore,
}

impl Shard {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let f = File::open(path)?;
        Self::from_file(f)
    }

    fn from_file(file: File) -> Result<Self, Error> {
        Ok(Self {
            file: Arc::new(ShardFile::from_file(file)?),
        })
    }

    pub fn docs(&self) -> DocStore {
        DocStore::new(Arc::clone(&self.file))
    }

    pub fn suffixes(&self) -> SuffixArrayStore {
        let sa_len = (self.file.header().sa.len / std::mem::size_of::<u32>() as u64) as u32;
        assert!(sa_len == self.file.header().docs.data.len as u32);
        SuffixArrayStore::new(Arc::clone(&self.file), sa_len)
    }
}
