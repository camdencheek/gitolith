use std::{collections::BTreeMap, fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Error};
use clap::Parser;
use gitserver3::shard::file::{CompoundSection, SimpleSection};
use serde_derive::Deserialize;
use std::io::{Seek, SeekFrom};

#[derive(Parser, Debug)]
pub struct Cli {
    pub input: PathBuf,
    pub output: PathBuf,
}

fn main() -> Result<(), Error> {
    let args = Cli::parse();
    let mut zoekt_shard_file = File::open(&args.input)?;
    let toc = ZoektTOC::from_file(&mut zoekt_shard_file)?;
    dbg!(toc);
    Ok(())
}

#[derive(Default, Debug)]
struct ZoektTOC {
    file_contents: CompoundSection,
    file_names: CompoundSection,
    repo_metadata: SimpleSection,
    repos: SimpleSection,
}

impl ZoektTOC {
    fn from_file(file: &mut File) -> Result<Self, Error> {
        file.seek(SeekFrom::End(-8))?;

        let offset = read_u32(file).context("read toc offset")?;
        let len = read_u32(file).context("read toc len")?;
        let toc_section = dbg!(SimpleSection {
            offset: offset as u64,
            len: len as u64,
        });

        file.seek(SeekFrom::Start(toc_section.offset))?;

        let mut toc = ZoektTOC::default();

        let section_count = read_u32(file)?;
        assert!(section_count == 0);

        while file.seek(SeekFrom::Current(0))? < toc_section.offset + toc_section.len {
            let tag = read_str(file)?;
            dbg!(&tag);
            let kind = read_uvarint(file)?;
            dbg!(&kind);
            match kind {
                0 => {
                    let offset = read_u32(file)? as u64;
                    let len = read_u32(file)? as u64;
                    match tag.as_str() {
                        "repoMetadata" => toc.repo_metadata = SimpleSection { offset, len },
                        "repos" => toc.repos = SimpleSection { offset, len },
                        _ => {}
                    }
                }
                1 | 2 => {
                    let offset1 = read_u32(file)? as u64;
                    let len1 = read_u32(file)? as u64;
                    let offset2 = read_u32(file)? as u64;
                    let len2 = read_u32(file)? as u64;
                    let section = CompoundSection {
                        data: SimpleSection {
                            offset: offset1,
                            len: len1,
                        },
                        offsets: SimpleSection {
                            offset: offset2,
                            len: len2,
                        },
                    };

                    match tag.as_str() {
                        "fileContents" => toc.file_contents = section,
                        "fileNames" => toc.file_names = section,
                        _ => {}
                    }
                }
                _ => unreachable!("kind should only be 0 or 1"),
            }
        }

        Ok(toc)
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, Error> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_be_bytes(b))
}

fn read_str<R: Read>(r: &mut R) -> Result<String, Error> {
    let slen = read_uvarint(r)?;
    let mut b = vec![0u8; slen as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8(b)?)
}

fn read_uvarint<R: Read>(r: &mut R) -> Result<u64, Error> {
    let mut x: u64 = 0;
    let mut s: usize = 0;
    for i in 0..10 {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        let b = b[0];

        if b < 0x80 {
            if i == 9 && b > 1 {
                return Err(anyhow::anyhow!("overflow"));
            }
            return Ok(x | ((b as u64) << s));
        }

        x |= ((b & 0x7f) as u64) << s;
        s += 7;
    }
    Err(anyhow::anyhow!("overflow"))
}

#[derive(Deserialize, Debug)]
struct RepoMetadata {
    #[serde(rename = "ID")]
    id: u32,

    #[serde(rename = "Name")]
    name: String,

    #[serde(rename = "URL")]
    url: String,

    #[serde(rename = "Source")]
    source: String,

    #[serde(rename = "Branches")]
    branches: Vec<RepoBranch>,

    #[serde(rename = "SubRepoMap")]
    sub_repo_map: BTreeMap<String, RepoMetadata>,

    #[serde(rename = "CommitURLTemplate")]
    commit_url_template: String,

    #[serde(rename = "FileURLTemplate")]
    file_url_template: String,

    #[serde(rename = "LineFragmentTemplate")]
    line_fragment_template: String,

    #[serde(rename = "RawConfig")]
    raw_config: BTreeMap<String, String>,

    #[serde(rename = "Rank")]
    rank: u16,

    #[serde(rename = "IndexOptions")]
    index_options: String,

    #[serde(rename = "HasSymbols")]
    has_symbols: bool,

    #[serde(rename = "Tombstone")]
    tombstone: bool,
    // TODO
    // latest_commit_date:
}

#[derive(Deserialize, Debug)]
struct RepoBranch {
    name: String,
    version: String,
}
