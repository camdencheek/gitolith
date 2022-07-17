#![allow(unused)]
pub mod shard;
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use regex::bytes::Regex;
use regex_syntax::ast::parse::Parser as AstParser;
use regex_syntax::hir::translate::Translator;
use shard::{RangesBuilder, Shard};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::ops::{Range, RangeInclusive};
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
    #[clap(long = "skip-index")]
    pub skip_index: bool,
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

    let mut query = args.query.as_str();
    if query.starts_with('/') && query.ends_with('/') && query.len() >= 2 {
        query = query.strip_prefix('/').unwrap().strip_suffix('/').unwrap();
        search_regex(s, query, args.skip_index)
    } else {
        search_literal(s, query)
    }
}

fn search_regex(s: Shard, query: &str, skip_index: bool) -> Result<(), Box<dyn Error>> {
    let re = Regex::new(query)?;

    if skip_index {
        let matches = s.search_skip_index(re);
        for m in matches {
            println!("Doc #{}", m.doc);
            let doc_content = s.doc_content(m.doc);
            for r in m.matched_ranges {
                println!(
                    "\t{}",
                    std::str::from_utf8(&doc_content[r.start as usize..r.end as usize])?
                );
            }
        }
    } else {
        let ast = AstParser::new().parse(query)?;
        let hir = Translator::new().translate(query, &ast)?;
        dbg!(&hir);
        let range_iters = RangesBuilder::from_hir(hir).build();

        // for doc in s.docs() {
        //     for mat in re.find_iter(doc.1) {
        //         println!("{:?}", std::str::from_utf8(mat.as_bytes())?);
        //     }
        // }
        for range_iter in range_iters {
            for range in range_iter {
                debug_range(&range);
                for suffix_idx in s.sa_range(range) {
                    println!(
                        "{}",
                        String::from_utf8(s.suffix(*suffix_idx)[..5].to_vec())?
                    );
                }
            }
        }
    }

    Ok(())
}

fn debug_range(r: &RangeInclusive<Vec<u8>>) {
    println!(
        "{}..{}",
        std::str::from_utf8(&r.start()).unwrap(),
        std::str::from_utf8(&r.end()).unwrap()
    );
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
