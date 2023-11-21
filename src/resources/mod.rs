use std::io::Write;

use tch::CModule;

macro_rules! import_model {
    ($name:ident, $path:literal) => {
        pub fn $name() -> CModule {
            let bytes = include_bytes!($path);
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            tmp.write_all(bytes).unwrap();
            let cmod = CModule::load(tmp.path()).unwrap();
            drop(tmp);
            cmod
        }
    };
}

import_model!(add_ten_cmodule, "add_ten.pt");
import_model!(busywork_cmodule, "busywork.pt");
