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
    pub output_shard: PathBuf,
    pub repo: PathBuf,
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
    let camdens = s.sa_prefixes(args.query.as_bytes());
    for camden in camdens {
        println!("{}", String::from_utf8(s.suffix(*camden)[..12].to_vec())?);
    }
    Ok(())
}

fn build_index(args: IndexArgs) -> Result<(), Box<dyn Error>> {
    let documents = WalkDir::new(args.repo)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| LazyFileReader {
            path: e.into_path(),
            f: None,
        });

    let s = Shard::new(0, &Path::new(&args.output_shard), documents)?;
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
