#![allow(unused)]
pub mod shard;
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use shard::Shard;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();
    match args.cmd {
        Command::Index(a) => build_index(a)?,
        Command::Search(a) => search(a)?,
    }
    Ok(())
}

fn search(args: SearchArgs) -> Result<(), Box<dyn Error>> {
    let s = Shard::open(0, &args.shard)?;
    let matches = s.sa_prefixes(args.query.as_bytes());
    for m in matches {
        let mut suffix = s.suffix(*m);
        suffix = &suffix[..usize::min(suffix.len(), 100)];

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

    let s = Shard::new(0, &Path::new(&output_shard), documents)?;
    Ok(())
}

fn build_string_index(output_shard: PathBuf, s: String) -> Result<(), Box<dyn Error>> {
    let documents = std::iter::once(s.as_bytes());
    let s = Shard::new(0, &Path::new(&output_shard), documents)?;
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
