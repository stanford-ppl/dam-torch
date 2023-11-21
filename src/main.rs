use std::path::PathBuf;

use clap::Parser;
use dam::{
    logging::{LogEvent, LogFilter},
    simulation::*,
    utility_contexts::*,
};
use model_context::AdapterType;
use tch::CModule;

use crate::{
    manager::*,
    model::Model,
    model_context::{InferenceData, ModelContext},
};

pub mod manager;
pub mod model;
pub mod model_context;

#[cfg(test)]
mod resources;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    pt_path: String,

    #[arg(long)]
    virtual_gpus: usize,

    #[arg(long)]
    physical_gpus: usize,

    #[arg(long)]
    batch_size: usize,

    #[arg(long)]
    num_batches: usize,

    #[arg(long)]
    db_name: String,

    #[arg(long, default_value = "mongodb://127.0.0.1:27017")]
    mongo_path: String,
}

type DTYPE = f32;
struct BasicAdapter {
    device: tch::Device,
}

impl AdapterType<DTYPE, DTYPE, CModule> for BasicAdapter {
    fn to_input(&self, input: Vec<DTYPE>) -> Vec<tch::Tensor> {
        vec![tch::Tensor::from_slice(&input)
            .to(self.device)
            .to_kind(tch::Kind::Float)]
    }

    fn from_output(&self, model_output: tch::Tensor) -> Vec<DTYPE> {
        model_output
            .iter::<f64>()
            .unwrap()
            .map(|x| x as DTYPE)
            .collect()
    }

    fn run(&self, model: &CModule, input: Vec<tch::Tensor>) -> tch::Tensor {
        model.forward_ts(&input).unwrap()
    }
}

fn main() {
    let args = Args::parse();

    const MODEL_LATENCY: u64 = 73;
    const MODEL_II: u64 = 1;

    let batch_size: usize = args.batch_size;
    let wait_latency: u64 = (args.batch_size * 2) as u64;
    let num_inputs: usize = args.num_batches * batch_size;
    if !tch::Cuda::is_available() {
        panic!("Cuda wasn't available!");
    }

    let mut ctx = ProgramBuilder::default();

    let (input_snd, input_rcv) = ctx.unbounded();
    ctx.add_child(GeneratorContext::new(
        || {
            // Generate one f64 at a time.
            (0..num_inputs).map(|x| x as DTYPE)
        },
        input_snd,
    ));

    let mut broadcaster = BroadcastContext::new(input_rcv);
    let models: Vec<_> = (0..args.physical_gpus)
        .map(|_: usize| JobRef::new(Model::from_path(PathBuf::from(args.pt_path.clone()))))
        .collect();

    for index in 0..args.virtual_gpus {
        let phys_gpu = index % args.physical_gpus;
        let device = tch::Device::Cuda(phys_gpu);
        let (lsnd, lrcv) = ctx.unbounded();
        broadcaster.add_target(lsnd);
        let (output_snd, output_rcv) = ctx.unbounded();

        ctx.add_child(ModelContext::new(
            lrcv,
            output_snd,
            None,
            wait_latency,
            batch_size,
            models[phys_gpu].clone(),
            BasicAdapter { device },
            MODEL_LATENCY,
            MODEL_II,
            device,
        ));
        ctx.add_child(ConsumerContext::new(output_rcv));
    }

    ctx.add_child(broadcaster);
    ctx.initialize(Default::default()).unwrap().run(
        RunOptionsBuilder::default()
            .mode(RunMode::FIFO)
            .log_filter(LogFilterKind::Blanket(LogFilter::Some(
                [InferenceData::NAME.to_owned()].into(),
            )))
            .logging(LoggingOptions::Mongo(
                MongoOptionsBuilder::default()
                    .db(args.db_name)
                    .uri(args.mongo_path)
                    .build()
                    .unwrap(),
            ))
            .build()
            .unwrap(),
    );
}
