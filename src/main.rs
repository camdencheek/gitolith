#![allow(unused)]
pub mod shard;
use shard::Shard;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

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
    Ok(())

    // let mut args = env::args();
    // args.next(); // skip first
    // let shard_path = args.next().expect("need shard path");

    // let mut documents: Vec<Vec<u8>> = Vec::new();
    // for arg in args {
    //     let mut buf: Vec<u8> = Vec::new();
    //     File::open(arg)?.read_to_end(&mut buf)?;
    //     documents.push(buf);
    // }

    // let s = Shard::new(0, &Path::new(&shard_path), documents.iter())?;
    // Ok(())
}
