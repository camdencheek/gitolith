#![allow(unused)]
pub mod shard;
use shard::Shard;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args();
    args.next(); // skip first
    let shard_path = args.next().expect("need shard path");

    let s = Shard::open(0, &Path::new(&shard_path))?;
    dbg!(s.header.content_ptr);
    dbg!(s.header.content_len);
    println!("{}", s.doc_content_range(0).end);
    println!("{}", s.doc_from_suffix(0));
    println!("{}", s.doc_from_suffix(3300));
    println!("{}", s.doc_from_suffix(3457));
    println!("{}", s.doc_from_suffix(3458));
    let camdens = s.sa_slice(b"camden"..b"camdeo");
    for camden in camdens {
        println!("{}", String::from_utf8(s.suffix(*camden)[..100].to_vec())?);
    }
    Ok(())

    // let mut args = env::args();
    // args.next(); // skip first
    // let shard_path = args.next().expect("need shard path");
    // let repo_path = args.next().expect("need repo path");

    // let documents = WalkDir::new(repo_path)
    //     .into_iter()
    //     .filter_map(|e| e.ok())
    //     .filter(|e| e.file_type().is_file())
    //     .map(|e| LazyFileReader {
    //         path: e.into_path(),
    //         f: None,
    //     });

    // let s = Shard::new(0, &Path::new(&shard_path), documents)?;
    // Ok(())
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
