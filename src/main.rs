#![allow(unused)]
pub mod shard;
pub mod suffix;
use shard::{Document, Shard};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args();
    args.next(); // skip first
    let shard_path = args.next().expect("need shard path");

    let mut documents: Vec<Vec<u8>> = Vec::new();
    for arg in args {
        let mut buf: Vec<u8> = Vec::new();
        File::open(arg)?.read_to_end(&mut buf)?;
        documents.push(buf);
    }

    let s = Shard::new(0, &Path::new(&shard_path), documents.iter())?;
    Ok(())
}
