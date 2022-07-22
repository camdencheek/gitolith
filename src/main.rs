pub mod shard;
use crate::shard::Shard;
use clap::{Parser, Subcommand};
use regex::bytes::Regex;
use std::error::Error;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

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

    let mut query = args.query.as_str();
    if query.starts_with('/') && query.ends_with('/') && query.len() >= 2 {
        query = query.strip_prefix('/').unwrap().strip_suffix('/').unwrap();
        search_regex(s, query, args.skip_index)
    } else {
        search_literal(s, query)
    }
}

fn list(args: ListArgs) -> Result<(), Box<dyn Error>> {
    let s = Shard::open(&args.shard)?;

    let handle = std::io::stdout().lock();
    let mut buf = std::io::BufWriter::new(handle);
    for doc in s.docs() {
        buf.write_fmt(format_args!("DocID: {}", doc.id))?;
        buf.write_fmt(format_args!(
            "===============\n{}\n==============\n",
            std::str::from_utf8(doc.content)?
        ))?;
    }
    Ok(())
}

fn search_regex(s: Shard, query: &str, skip_index: bool) -> Result<(), Box<dyn Error>> {
    let re = Regex::new(query)?;

    let handle = std::io::stdout().lock();
    let mut buf = std::io::BufWriter::new(handle);

    if skip_index {
        let matches = s.search_skip_index(re);
        for m in matches {
            println!("Doc #{}", m.doc.id);
            for r in m.matched_ranges {
                println!(
                    "\t{}",
                    std::str::from_utf8(&m.doc.content[r.start as usize..r.end as usize])?
                );
            }
        }
    } else {
        let matches = s.search(&re);
        for m in matches {
            buf.write_fmt(format_args!("DocID: {}\n", m.doc.id))?;
            for range in m.matches {
                buf.write(&m.doc.content[range.start as usize..range.end as usize])?;
                buf.write(b"\n")?;
            }
        }
    }

    Ok(())
}

fn search_literal(s: Shard, query: &str) -> Result<(), Box<dyn Error>> {
    let b = query.as_bytes();
    dbg!(&b);
    let matches = s.sa_prefixes(b);
    for m in matches {
        let mut suffix = s.suffix(*m);
        suffix = &suffix[..usize::min(suffix.len(), query.len())];

        println!("{}", String::from_utf8(suffix.to_vec())?);
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
        .filter(|e| e.file_type().is_file())
        .map(|e| LazyFileReader {
            path: e.into_path(),
            f: None,
        });

    let _s = Shard::new(0, &Path::new(&output_shard), documents)?;
    Ok(())
}

fn build_string_index(output_shard: PathBuf, s: String) -> Result<(), Box<dyn Error>> {
    let documents = std::iter::once(s.as_bytes());
    let _s = Shard::new(0, &Path::new(&output_shard), documents)?;
    Ok(())
}

struct LazyFileReader {
    path: PathBuf,
    f: Option<File>,
}

impl Read for LazyFileReader {
    fn read(&mut self, mut buf: &mut [u8]) -> Result<usize, std::io::Error> {
        match &mut self.f {
            Some(file) => file.read(&mut buf),
            None => {
                self.f = Some(File::open(&self.path)?);
                self.read(&mut buf)
            }
        }
    }
}
