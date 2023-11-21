use std::path::PathBuf;

use fxhash::FxHashMap;
use tch::CModule;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModuleError {
    #[error("Read Error")]
    IO(#[from] std::io::Error),

    #[error("Module Loading Error")]
    Tch(#[from] tch::TchError),
}

pub fn load_modules(path: PathBuf) -> Result<FxHashMap<PathBuf, CModule>, ModuleError> {
    let entries = std::fs::read_dir(path)?;
    let mut result = FxHashMap::default();

    for entry in entries {
        let path = entry?.path();
        let module = CModule::load(&path)?;
        result.insert(path, module);
    }

    Ok(result)
}
