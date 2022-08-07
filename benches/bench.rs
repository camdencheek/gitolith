use std::{
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::Error;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gitserver3::{
    cache::new_cache,
    search::search_regex,
    shard::{builder::ShardBuilder, cached::CachedShard, Shard, ShardID},
};
use walkdir::WalkDir;

fn search_shard(shard: CachedShard, query: &str) -> usize {
    rayon::in_place_scope_fifo(|s| -> usize {
        search_regex(shard, query, false, s)
            .unwrap()
            .map(|dm| dm.matches.len())
            .sum()
    })
}

fn maybe_build_directory_index(output_shard: PathBuf, dir: PathBuf) -> Result<(), Error> {
    if !output_shard.exists() {
        build_directory_index(output_shard, dir)
    } else {
        Ok(())
    }
}

fn build_directory_index(output_shard: PathBuf, dir: PathBuf) -> Result<(), Error> {
    let documents = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file());

    let mut builder = ShardBuilder::new(&Path::new(&output_shard))?;
    for entry in documents {
        let f = File::open(entry.path())?;
        let l = f.metadata()?.len();
        if l > (2 << 20) {
            continue;
        }
        builder.add_doc(f)?;
    }
    builder.build()?;
    Ok(())
}

fn criterion_benchmark(c: &mut Criterion) {
    maybe_build_directory_index(
        "/tmp/shardlinux".into(),
        "/Users/camdencheek/src/linux".into(),
    )
    .unwrap();
    let shard = Shard::open(&Path::new("/tmp/shardlinux")).unwrap();
    let cache = new_cache(256 * 1024 * 1024);
    let cached_shard = CachedShard::new(ShardID(0), shard, cache);

    for query in &[
        "torvalds",
        "(?i)torvalds",
        "module_put_and_pthread",
        "module_put_.*_pthread",
        r"\w+\(char \*\w+\)",
    ] {
        c.bench_function(&query, |b| {
            b.iter(|| search_shard(cached_shard.clone(), &query))
        });
    }

    c.bench_function("combined case insensitive", |b| {
        b.iter(|| {
            for query in &[
                "(?i)torvalds",
                "(?i)rq_qos_id",
                "(?i)invalidate_bdev",
                "(?i)crc_itu_t",
                "(?i)mmrdpcstx1",
            ] {
                search_shard(cached_shard.clone(), &query);
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
