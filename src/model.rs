use std::path::PathBuf;

use tempfile::NamedTempFile;

struct Model<T> {
    path: Option<PathBuf>,
    stash: Option<NamedTempFile>,
    module: Option<T>,
}
