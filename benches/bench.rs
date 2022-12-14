use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::Error;
use criterion::{criterion_group, criterion_main, Criterion};
use gitserver3::{
    search::search_regex,
    shard::{builder::ShardBuilder, Shard},
};
use regex::bytes::Regex;
use walkdir::WalkDir;

fn search_shard(shard: Shard, query: &str) -> usize {
    let re = Regex::new(query).unwrap();
    search_regex(shard, re, false)
        .map(|dm| dm.matches.len())
        .sum()
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
        let mut f = File::open(entry.path())?;
        let l = f.metadata()?.len();
        if l > (2 << 20) {
            continue;
        }
        let mut v = Vec::new();
        f.read_to_end(&mut v)?;
        builder.add_doc(entry.path().to_string_lossy().into(), v)?;
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
    let shard = Shard::open_default(&Path::new("/tmp/shardlinux")).unwrap();

    for query in &[
        "torvalds",
        "(?i)torvalds",
        "module_put_and_pthread",
        "module_put_.*_pthread",
        r"\w+\(char \*\w+\)",
    ] {
        c.bench_function(&query, |b| b.iter(|| search_shard(shard.clone(), &query)));
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
                search_shard(shard.clone(), &query);
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
