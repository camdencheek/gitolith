#![allow(unused)]

use clap::{Parser, Subcommand};
use regex::Regex;
use search::search_regex;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

mod search;
mod shard;
use shard::builder::ShardBuilder;
use shard::Shard;

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
}

#[derive(Parser, Debug)]
pub struct ListArgs {
    pub shard: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();
    match args.cmd {
        Command::Index(a) => build_index(a)?,
        Command::Search(a) => search(a)?,
        Command::List(a) => list(a)?,
    }
    Ok(())
}

fn search(args: SearchArgs) -> Result<(), Box<dyn Error>> {
    let s = Shard::open(&args.shard)?;

    let handle = std::io::stdout().lock();
    let mut buf = std::io::BufWriter::new(handle);

    for doc_match in search_regex(&s, &args.query, args.skip_index)? {
        let doc_match = doc_match?;
        buf.write_fmt(format_args!("{:?}:\n", doc_match.id))?;
        for r in doc_match.matches {
            buf.write_fmt(format_args!(
                "{}\n",
                std::str::from_utf8(&doc_match.content[r.start as usize..r.end as usize])?,
            ))?;
        }
    }
    Ok(())
}

fn list(args: ListArgs) -> Result<(), Box<dyn Error>> {
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

fn build_index(args: IndexArgs) -> Result<(), Box<dyn Error>> {
    if let Some(dir) = args.dir {
        return build_directory_index(args.output_shard, dir);
    } else if let Some(s) = args.string {
        return build_string_index(args.output_shard, s);
    } else {
        panic!("must specify a directory or a string to index")
    }
}

fn build_directory_index(output_shard: PathBuf, dir: PathBuf) -> Result<(), Box<dyn Error>> {
    let documents = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file());

    let mut builder = ShardBuilder::new(&Path::new(&output_shard))?;
    for entry in documents {
        let f = File::open(entry.path())?;
        builder.add_doc(f)?;
    }
    builder.build()?;
    Ok(())
}

fn build_string_index(output_shard: PathBuf, s: String) -> Result<(), Box<dyn Error>> {
    let mut builder = ShardBuilder::new(&Path::new(&output_shard))?;
    builder.add_doc(s.as_bytes())?;
    builder.build()?;
    Ok(())
}
