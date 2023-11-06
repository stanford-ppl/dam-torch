use std::io::Write;

use tch::CModule;

pub fn add_ten_cmodule() -> CModule {
    let bytes = include_bytes!("add_ten.pt");
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(bytes).unwrap();
    let cmod = CModule::load(tmp.path()).unwrap();
    drop(tmp);
    cmod
}
