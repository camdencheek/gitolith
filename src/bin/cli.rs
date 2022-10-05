// #![allow(unused)]

use anyhow::Error;
use clap::{Parser, Subcommand};
use gitserver3::shard::cached_file::CachedShardFile;
use gitserver3::shard::docs::{ContentIdx, DocID};
use gitserver3::shard::file::{ShardFile, ShardStore};
use gitserver3::shard::suffix::{SuffixArrayStore, SuffixBlockID, SuffixIdx};
use regex::bytes::Regex;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;

use gitserver3::cache;
use gitserver3::search::search_regex;
use gitserver3::shard::builder::ShardBuilder;
use gitserver3::shard::{Shard, ShardID};

#[derive(Parser, Debug)]
pub struct Cli {
    #[clap(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    Index(IndexArgs),
    Search(SearchArgs),
    Debug(DebugArgs),
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
    #[clap(long = "limit", short = 'l')]
    pub limit: Option<usize>,
    #[clap(long = "cache-size")]
    pub cache_size: Option<String>,
    #[clap(long = "no-cache")]
    pub no_cache: bool,
}

#[derive(Parser, Debug)]
pub struct DebugArgs {
    pub shard: PathBuf,
    #[clap(subcommand)]
    pub debug_command: DebugCommand,
}

#[derive(Subcommand, Debug)]
pub enum DebugCommand {
    Docs,
    DocContent(DocIDArg),
    DocBounds(DocIDArg),
    SuffixContentIdx(SuffixIDArg),
    Content(ContentIDArg),
    SuffixBlock(SuffixIDArg),
    Block(BlockIDArg),
}

#[derive(Parser, Debug)]
pub struct DocIDArg {
    id: u32,
}

#[derive(Parser, Debug)]
pub struct BlockIDArg {
    id: u32,
}

#[derive(Parser, Debug)]
pub struct SuffixIDArg {
    id: u32,
}

#[derive(Parser, Debug)]
pub struct ContentIDArg {
    id: u32,
}

fn main() -> Result<(), Error> {
    let args = Cli::parse();
    match args.cmd {
        Command::Index(a) => build_index(a)?,
        Command::Search(a) => search(a)?,
        Command::Debug(a) => debug(a)?,
    }
    Ok(())
}

fn search(args: SearchArgs) -> Result<(), Error> {
    let cache_size: u64 = match args.cache_size {
        Some(s) => bytefmt::parse(&s).expect("failed to parse cache size"),
        None => 256 * 1024 * 1024,
    };
    let csf: ShardStore = if args.no_cache {
        Arc::new(ShardFile::from_file(File::open(args.shard)?)?)
    } else {
        Arc::new(CachedShardFile::new(
            ShardID(0),
            cache::new_cache(cache_size),
            ShardFile::from_file(File::open(args.shard)?)?,
        ))
    };
    let cs = Shard::from_store(csf);
    let re = Regex::new(&args.query)?;

    for i in 0..args.repeat {
        let start = Instant::now();

        let handle = std::io::stdout().lock();
        let mut buf = std::io::BufWriter::new(handle);

        let mut count = 0;
        let mut limit = args.limit.unwrap_or(usize::MAX);

        for mut doc_match in search_regex(cs.clone(), re.clone(), args.skip_index) {
            doc_match.matches.truncate(limit);
            limit -= doc_match.matches.len();
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
            if limit == 0 {
                break;
            }
        }

        buf.write_fmt(format_args!(
            "Iter: {}, Searched: , Match Count: {}, Elapsed: {:3.2?}\n",
            i,
            // bytefmt::format(content_size),
            count,
            start.elapsed()
        ))?;

        buf.flush()?;
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
        let mut f = File::open(entry.path())?;
        let l = f.metadata()?.len();
        if l > (2 << 20) {
            println!("skipping file {:?} with size {}", entry.path(), l);
            continue;
        }

        let mut v = Vec::new();
        f.read_to_end(&mut v)?;
        builder.add_doc(entry.path().to_string_lossy().into(), v)?;
    }
    builder.build()?;
    Ok(())
}

fn build_string_index(output_shard: PathBuf, s: String) -> Result<(), Error> {
    let mut builder = ShardBuilder::new(Path::new(&output_shard))?;
    builder.add_doc("sample file name".into(), s.as_bytes().to_vec())?;
    builder.build()?;
    Ok(())
}

fn debug(args: DebugArgs) -> Result<(), Error> {
    let shard = Shard::open_default(&args.shard)?;

    match args.debug_command {
        DebugCommand::Docs => print_docs(shard)?,
        DebugCommand::DocContent(doc_id) => print_doc_content(shard, doc_id.id)?,
        DebugCommand::DocBounds(doc_id) => print_doc_bounds(shard, doc_id.id)?,
        DebugCommand::SuffixContentIdx(suffix_id) => print_suffix_content_idx(shard, suffix_id.id)?,
        DebugCommand::Content(content_idx) => print_content(shard, content_idx.id)?,
        DebugCommand::SuffixBlock(suffix_id) => print_suffix_block(suffix_id.id)?,
        DebugCommand::Block(block_id) => print_block(shard, block_id.id)?,
    }

    Ok(())
}

fn print_docs(shard: Shard) -> Result<(), Error> {
    let doc_ends = shard.docs().read_doc_ends()?;
    for doc in doc_ends.iter_docs() {
        println!(
            "ID: {}, [{}..{}]",
            u32::from(doc),
            u32::from(doc_ends.doc_start(doc)),
            u32::from(doc_ends.doc_end(doc))
        );
    }
    Ok(())
}

fn print_doc_content(shard: Shard, doc_id: u32) -> Result<(), Error> {
    let doc_ends = shard.docs().read_doc_ends()?;
    let content = shard.docs().read_content(DocID(doc_id), &doc_ends)?;
    let mut handle = std::io::stdout().lock();
    handle.write(&content)?;
    Ok(())
}

fn print_doc_bounds(shard: Shard, doc_id: u32) -> Result<(), Error> {
    let doc_ends = shard.docs().read_doc_ends()?;
    println!(
        "[{}..{}]",
        u32::from(doc_ends.doc_start(DocID(doc_id))),
        u32::from(doc_ends.doc_end(DocID(doc_id)))
    );
    Ok(())
}

fn print_suffix_content_idx(shard: Shard, suffix_id: u32) -> Result<(), Error> {
    let (block_id, offset) = SuffixArrayStore::block_id_for_suffix(SuffixIdx(suffix_id));
    let block = shard.suffixes().read_block(block_id)?;
    println!("{}", u32::from(block.0[offset]),);
    Ok(())
}

fn print_content(shard: Shard, content_id: u32) -> Result<(), Error> {
    let docs = shard.docs();
    let doc_ends = shard.docs().read_doc_ends()?;
    let doc_id = doc_ends.find(ContentIdx(content_id));
    let doc_start = doc_ends.doc_start(doc_id);
    let offset = content_id - u32::from(doc_start);
    let doc_content = docs.read_content(doc_id, &doc_ends)?;
    let suffix = &doc_content[offset as usize..];
    let mut handle = std::io::stdout().lock();
    handle.write(&suffix)?;
    Ok(())
}

fn print_block(shard: Shard, suffix_block_id: u32) -> Result<(), Error> {
    let block = shard
        .suffixes()
        .read_block(SuffixBlockID(suffix_block_id))?;
    for suffix in block.0.iter().cloned() {
        println!("{}", u32::from(suffix))
    }
    Ok(())
}

fn print_suffix_block(suffix_idx: u32) -> Result<(), Error> {
    let (block_id, offset) = SuffixArrayStore::block_id_for_suffix(SuffixIdx(suffix_idx));
    println!("{}, offset {}", u32::from(block_id), offset);
    Ok(())
}
