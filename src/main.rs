#![feature(core_intrinsics)]
#![feature(const_char_convert)]
mod cache;
mod shard;
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
    Ok(())
}
