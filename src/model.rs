use std::path::PathBuf;

use tempfile::NamedTempFile;

#[derive(Debug)]
pub struct Model<T> {
    path: Option<PathBuf>,

    stash: Option<NamedTempFile>,

    module: Option<T>,
}

impl Model<tch::CModule> {
    pub fn from_path(path: PathBuf) -> Self {
        Self {
            path: Some(path),
            stash: None,
            module: None,
        }
    }
}

impl<T> Model<T> {
    pub fn from_module(module: T) -> Self {
        Self {
            path: None,
            stash: None,
            module: Some(module),
        }
    }
}

impl super::manager::Job for Model<tch::CModule> {
    fn load(&mut self, device: tch::Device) {
        assert!(self.module.is_none());

        // load from stash
        if let Some(file) = &self.stash {
            self.module = Some(tch::CModule::load_on_device(file.path(), device).unwrap());
            return;
        }

        // load from file
        if let Some(path) = &self.path {
            // load from stash
            self.module = Some(tch::CModule::load_on_device(path, device).unwrap());
            return;
        }

        panic!("Cannot load module; it is neither stashed nor in a file.")
    }

    fn stash(&mut self) {
        let module = self
            .module
            .take()
            .expect("Cannot stash module: it wasn't loaded!");

        let path = self.stash.get_or_insert_with(make_pytorch_tempfile).path();

        module.save(path).unwrap();
    }
}

fn make_pytorch_tempfile() -> NamedTempFile {
    tempfile::Builder::new().suffix(".pt").tempfile().unwrap()
}

#[cfg(test)]
mod test {

    use crate::{manager::Job, resources::add_ten_cmodule};

    use super::Model;

    #[test]
    fn basic_functionality() {
        let mut model = Model::from_module(add_ten_cmodule());
        println!("Model loaded");
        model.stash();
        println!("Model stashed");
        model.load(tch::Device::cuda_if_available());
        println!("Model re-loaded");
        let result = model.module.map(|module| {
            module
                .forward_ts(&[tch::Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4])])
                .unwrap()
        });

        dbg!(result);
    }
}
