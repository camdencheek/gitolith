// #![allow(unused)]

use anyhow::Error;
use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

use gitserver3::cache;
use gitserver3::search::search_regex;
use gitserver3::shard::builder::ShardBuilder;
use gitserver3::shard::cached::CachedShard;
use gitserver3::shard::{Shard, ShardID};
use gitserver3::strcmp;

#[derive(Parser, Debug)]
pub struct Cli {
    #[clap(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    Index(IndexArgs),
    Search(SearchArgs),
    List(ListArgs),
}

#[derive(Parser, Debug)]
pub struct IndexArgs {
    #[clap(short = 'o')]
    pub output_shard: PathBuf,
    #[clap(long = "dir")]
    pub dir: Option<PathBuf>,
    #[clap(long = "str")]
    pub string: Option<String>,
}

#[derive(Parser, Debug)]
pub struct SearchArgs {
    pub shard: PathBuf,
    pub query: String,
    #[clap(long = "skip-index")]
    pub skip_index: bool,
    #[clap(long = "count-only")]
    pub count_only: bool,
    #[clap(long = "repeat", short = 'r', default_value = "1")]
    pub repeat: usize,
    #[clap(long = "cache-size")]
    pub cache_size: Option<String>,
}

#[derive(Parser, Debug)]
pub struct ListArgs {
    pub shard: PathBuf,
}

fn main() -> Result<(), Error> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()?;

    let args = Cli::parse();
    match args.cmd {
        Command::Index(a) => build_index(a)?,
        Command::Search(a) => search(a)?,
        Command::List(a) => list(a)?,
    }
    Ok(())
}

fn search(args: SearchArgs) -> Result<(), Error> {
    let cache_size: u64 = match args.cache_size {
        Some(s) => bytefmt::parse(&s).expect("failed to parse cache size"),
        None => 256 * 1024 * 1024,
    };
    let s = Shard::open(&args.shard)?;
    let content_size = s.header.content_len;
    let c = cache::new_cache(cache_size); // 4 GiB
    let cs = CachedShard::new(ShardID(0), s, c);

    for i in 0..args.repeat {
        let start = Instant::now();

        rayon::in_place_scope_fifo(|s| -> Result<(), Error> {
            let handle = std::io::stdout().lock();
            let mut buf = std::io::BufWriter::new(handle);

            let mut count = 0;

            for doc_match in search_regex(cs.clone(), &args.query, args.skip_index, s)? {
                if !args.count_only {
                    buf.write_fmt(format_args!("{:?}:\n", doc_match.id))?;
                }
                for r in doc_match.matches {
                    count += 1;

                    if !args.count_only {
                        buf.write_fmt(format_args!(
                            "{}\n",
                            String::from_utf8_lossy(
                                &doc_match.content[r.start as usize..r.end as usize]
                            ),
                        ))?;
                    }
                }
            }
            buf.flush()?;

            println!(
                "Iter: {}, Searched: {}, Match Count: {}, Elapsed: {:?}",
                i,
                bytefmt::format(content_size),
                count,
                start.elapsed()
            );
            Ok(())
        })?;
    }

    Ok(())
}

fn list(args: ListArgs) -> Result<(), Error> {
    let s = Shard::open(&args.shard)?;

    let handle = std::io::stdout().lock();
    let mut buf = std::io::BufWriter::new(handle);
    let doc_ends = s.docs.read_doc_ends()?;
    for doc_id in s.docs.doc_ids() {
        buf.write_fmt(format_args!("DocID: {}\n", u32::from(doc_id)))?;
        buf.write_fmt(format_args!(
            "===============\n{}\n===============\n",
            std::str::from_utf8(&s.docs.read_content(doc_id, &doc_ends)?)?
        ))?;
    }
    Ok(())
}

fn build_index(args: IndexArgs) -> Result<(), Error> {
    if let Some(dir) = args.dir {
        build_directory_index(args.output_shard, dir)
    } else if let Some(s) = args.string {
        build_string_index(args.output_shard, s)
    } else {
        panic!("must specify a directory or a string to index")
    }
}

fn build_directory_index(output_shard: PathBuf, dir: PathBuf) -> Result<(), Error> {
    let documents = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file());

    let mut builder = ShardBuilder::new(Path::new(&output_shard))?;
    for entry in documents {
        let f = File::open(entry.path())?;
        let l = f.metadata()?.len();
        if l > (2 << 20) {
            println!("skipping file {:?} with size {}", entry.path(), l);
            continue;
        }
        builder.add_doc(f)?;
    }
    builder.build()?;
    Ok(())
}

fn build_string_index(output_shard: PathBuf, s: String) -> Result<(), Error> {
    let mut builder = ShardBuilder::new(Path::new(&output_shard))?;
    builder.add_doc(s.as_bytes())?;
    builder.build()?;
    Ok(())
}
