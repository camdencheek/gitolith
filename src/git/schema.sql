CREATE TABLE repos (
    id INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    git_dir TEXT NOT NULL
);

CREATE TABLE indexed_blobs (
    shard_id INTEGER NOT NULL,
    doc_id INTEGER NOT NULL,
    repo_id INTEGER NOT NULL,
    blob_oid BLOB NOT NULL,
    filenames BLOB NOT NULL,

    PRIMARY KEY (shard_id, doc_id)
);
