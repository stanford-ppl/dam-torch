pub mod manager;
pub mod model;
pub mod model_context;

#[cfg(test)]
mod tests {
    use tch::{Cuda, Tensor};

    #[test]
    fn it_works() {
        let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
        let t = t * 2;
        t.print();
    }

    #[test]
    fn cuda_available() {
        println!("Cuda Available: {}", Cuda::is_available());
    }
}
