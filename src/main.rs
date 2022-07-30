#![allow(unused)]

use clap::{Parser, Subcommand};
use regex::Regex;
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

    let mut query = args.query.as_str();
    if query.starts_with('/') && query.ends_with('/') && query.len() >= 2 {
        query = query.strip_prefix('/').unwrap().strip_suffix('/').unwrap();
        search_regex(s, query, args.skip_index)?;
    } else {
        // search_literal(s, query)
    };
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

fn search_regex(s: Shard, query: &str, skip_index: bool) -> Result<(), Box<dyn Error>> {
    let re = Regex::new(query)?;
    let ast = regex_syntax::ast::parse::Parser::new()
        .parse(re.as_str())
        .expect("regex str failed to parse as AST");
    let hir = regex_syntax::hir::translate::Translator::new()
        .translate(re.as_str(), &ast)
        .expect("regex str failed to parse for translator");

    let handle = std::io::stdout().lock();
    let mut buf = std::io::BufWriter::new(handle);

    let extracted = search::regex::extract_regex_literals(hir);
    match extracted {
        search::regex::ExtractedRegexLiterals::None => {
            println!("No literals extracted")
        }
        search::regex::ExtractedRegexLiterals::Exact(set) => {
            let mut start = Vec::new();
            let mut end = Vec::new();
            for i in 0..dbg!(set.len()) {
                set.write_state_to(i, &mut start, &mut end);
                buf.write(b"Exact: ")?;
                buf.write(&start)?;
                buf.write(b"..=")?;
                buf.write(&end)?;
                buf.write(b"\n")?;
                start.clear();
                end.clear();
            }
        }
        search::regex::ExtractedRegexLiterals::Inexact(all) => {
            for (i, set) in all.iter().enumerate() {
                buf.write_fmt(format_args!("Inexact set #{}", i))?;
                let mut start = Vec::new();
                let mut end = Vec::new();
                for i in 0..set.len() {
                    set.write_state_to(i, &mut start, &mut end);
                    buf.write(b"Exact: ")?;
                    buf.write(&start)?;
                    buf.write(b"..=")?;
                    buf.write(&end)?;
                    buf.write(b"\n")?;
                    start.clear();
                    end.clear();
                }
            }
        }
    }
    // let handle = std::io::stdout().lock();
    // let mut buf = std::io::BufWriter::new(handle);

    // if skip_index {
    //     let matches = s.search_skip_index(re);
    //     for m in matches {
    //         println!("Doc #{}", m.doc.id);
    //         for r in m.matched_ranges {
    //             println!(
    //                 "\t{}",
    //                 std::str::from_utf8(&m.doc.content[r.start as usize..r.end as usize])?
    //             );
    //         }
    //     }
    // } else {
    // let matches = s.search(&re);
    // for m in matches {
    //     buf.write_fmt(format_args!("DocID: {}\n", m.doc.id))?;
    //     for range in m.matches {
    //         buf.write(&m.doc.content[range.start as usize..range.end as usize])?;
    //         buf.write(b"\n")?;
    //     }
    // }
    // }

    Ok(())
}

// fn search_literal(s: Shard, query: &str) -> Result<(), Box<dyn Error>> {
//     let b = query.as_bytes();
//     dbg!(&b);
//     let matches = s.sa().find(b..=b);
//     for m in matches {
//         let mut suffix = s.suffix(m);
//         suffix = &suffix[..usize::min(suffix.len(), query.len())];

//         println!("{}", String::from_utf8(suffix.to_vec())?);
//     }
//     Ok(())
// }

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
