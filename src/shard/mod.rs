pub mod builder;
pub mod cached_file;
pub mod docs;
pub mod file;
pub mod suffix;
use crate::cache;

use self::{
    cached_file::CachedShardFile,
    file::{ShardFile, ShardStore},
};
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
    pub fn from_store(store: ShardStore) -> Self {
        Self { file: store }
    }

    pub fn open_default(path: &Path) -> Result<Self, Error> {
        let f = ShardFile::from_file(File::open(path)?)?;
        let c = cache::new_cache(512 * 1024 * 1024);
        let cached_store = Arc::new(CachedShardFile::new(ShardID(0), c, f));
        Ok(Self { file: cached_store })
    }

    pub fn docs(&self) -> DocStore {
        DocStore::new(Arc::clone(&self.file))
    }

    pub fn suffixes(&self) -> SuffixArrayStore {
        SuffixArrayStore::new(Arc::clone(&self.file))
    }
}

pub type Trigram = [u8; 3];
