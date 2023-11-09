use std::collections::{HashMap, HashSet};

use dam::{
    channel::ChannelID,
    context::{self, ContextSummary, ExplicitConnections, ProxyContext},
    context_tools::*,
    logging::{copy_log, initialize_log},
    structures::{Identifiable, Identifier, ParentView, Time, TimeViewable, VerboseIdentifier},
    types::Cleanable,
};

use crate::{
    manager::{job_manager, Job, JobRef},
    model::Model,
};

pub trait AdapterType<InType, OutType, ModelType> {
    // Converts between inputs
    fn to_input(&self, input: Vec<InType>) -> Vec<tch::Tensor>;

    fn from_output(&self, model_output: tch::Tensor) -> Vec<OutType>;

    fn run(&self, model: &ModelType, input: Vec<tch::Tensor>) -> tch::Tensor;
}

pub struct ModelContext<T, RecvType: DAMType, SendType: DAMType, AT> {
    batcher: ProxyContext<InputBatcher<RecvType>>,
    engine: ProxyContext<InferenceEngine<T, RecvType, SendType, AT>>,

    id: Identifier,
    input_channel_id: ChannelID,
    output_channel_id: ChannelID,
}

impl<T, RT, ST, AT> ModelContext<T, RT, ST, AT>
where
    RT: DAMType,
    ST: DAMType,
    Model<T>: Job,
    InputBatcher<RT>: Context,
    InferenceEngine<T, RT, ST, AT>: Context,
{
    pub fn new(
        input: Receiver<RT>,
        output: Sender<ST>,
        queue_depth: Option<usize>,
        wait_latency: u64,
        max_batch_size: usize,
        model: Model<T>,
        adapter: AT,
        latency: u64,
        initiation_interval: u64,
        device: tch::Device,
    ) -> Self {
        let (comm_snd, comm_rcv) = match queue_depth {
            Some(size) => crossbeam::channel::bounded(size),
            None => crossbeam::channel::unbounded(),
        };
        let input_channel_id = input.id();
        let output_channel_id = output.id();
        let ctx = Self {
            batcher: InputBatcher {
                wait_latency,
                max_batch_size,
                input,
                communicator: comm_snd,
                context_info: Default::default(),
            }
            .into(),
            engine: InferenceEngine {
                communicator: comm_rcv,
                model: JobRef::new(model),
                output,
                adapter,
                latency,
                device,
                initiation_interval,
                context_info: Default::default(),
            }
            .into(),
            id: Identifier::new(),
            input_channel_id,
            output_channel_id,
        };
        ctx.batcher.input.attach_receiver(&*ctx.batcher);
        ctx.engine.output.attach_sender(&*ctx.engine);
        ctx
    }
}

impl<T, RT: DAMType, ST: DAMType, AT> Identifiable for ModelContext<T, RT, ST, AT> {
    fn id(&self) -> Identifier {
        self.id
    }

    fn name(&self) -> String {
        "ModelContext".to_string()
    }
}

impl<T: Send + Sync, RT: DAMType, ST: DAMType, AT> TimeViewable for ModelContext<T, RT, ST, AT>
where
    InputBatcher<RT>: Context,
    InferenceEngine<T, RT, ST, AT>: Context,
{
    fn view(&self) -> dam::structures::TimeView {
        ParentView {
            child_views: vec![self.batcher.view(), self.engine.view()],
        }
        .into()
    }
}

impl<T: Sync + Send, RT: DAMType, ST: DAMType, AdapterType> Context
    for ModelContext<T, RT, ST, AdapterType>
where
    InputBatcher<RT>: Context,
    InferenceEngine<T, RT, ST, AdapterType>: Context,
{
    fn run(&mut self) {
        let read_log = copy_log();
        let write_log = copy_log();
        std::thread::scope(|s| {
            s.spawn(|| {
                if let Some(mut logger) = read_log {
                    logger.id = self.batcher.id();
                    initialize_log(logger);
                }
                self.batcher.run();
                self.batcher.cleanup();
            });
            s.spawn(|| {
                if let Some(mut logger) = write_log {
                    logger.id = self.engine.id();
                    initialize_log(logger);
                }
                self.engine.run();
                self.engine.cleanup();
            });
        });
    }

    fn ids(&self) -> HashMap<VerboseIdentifier, HashSet<VerboseIdentifier>> {
        let mut base: HashMap<_, _> = [(
            self.verbose(),
            [self.batcher.verbose(), self.engine.verbose()].into(),
        )]
        .into();
        base.extend(self.batcher.ids());
        base.extend(self.engine.ids());
        base
    }

    fn edge_connections(&self) -> Option<ExplicitConnections> {
        Some(
            [(
                self.id,
                vec![(
                    [self.input_channel_id].into(),
                    [self.output_channel_id].into(),
                )],
            )]
            .into(),
        )
    }

    fn summarize(&self) -> context::ContextSummary {
        ContextSummary {
            id: self.verbose(),
            time: self.view(),
            children: vec![self.batcher.summarize(), self.engine.summarize()],
        }
    }
}

/// Groups inputs up into batches, with a pre-defined max size.
/// Do not use by itself; the struct is made public in order to keep type bounds cleaner for [ModelContext]
#[context_macro]

pub struct InputBatcher<RT: DAMType> {
    wait_latency: u64,
    max_batch_size: usize,

    input: Receiver<RT>,
    communicator: crossbeam::channel::Sender<(Time, Vec<RT>)>,
}

impl<RT: DAMType> Context for InputBatcher<RT> {
    fn run(&mut self) {
        'main: loop {
            let mut batch: Vec<RT> = vec![];
            let mut batch_start: Option<Time> = None;
            loop {
                match self.input.peek_next(&self.time) {
                    Ok(ChannelElement { time, data }) => {
                        // If we're too late. Send what we have and then continue.
                        if let Some(start) = batch_start {
                            if start + self.wait_latency < time {
                                let transmit_time = start + self.wait_latency;
                                let _ = self.communicator.send((transmit_time, batch));
                                continue 'main;
                            }
                        }

                        // If this is the first item in the batch, then we need to mark the start of the batch.
                        if let None = batch_start {
                            batch_start = Some(time);
                        }

                        batch.push(data);

                        // Pop off the peeked element.
                        let _ = self.input.dequeue(&self.time);

                        if batch.len() == self.max_batch_size {
                            // our batch is full, so we should send it along.
                            let transmit_time = self.time.tick();
                            let _ = self.communicator.send((transmit_time, batch));
                            continue 'main;
                        }
                    }
                    Err(_) if batch.is_empty() => {
                        return;
                    }
                    Err(_) => {
                        // The batch isn't empty, so we need to send it along as well.
                        let transmit_time = batch_start.unwrap() + self.wait_latency;
                        let _ = self.communicator.send((transmit_time, batch));
                        return;
                    }
                }
            }
        }
    }
}

/// Runs the actual inference, and then figures out its output time.
/// Do not use by itself; the struct is made public in order to keep type bounds cleaner for [ModelContext]
#[context_macro]
pub struct InferenceEngine<T, RT: DAMType, ST: DAMType, AT> {
    communicator: crossbeam::channel::Receiver<(Time, Vec<RT>)>,

    model: JobRef<Model<T>>,
    output: Sender<ST>,
    adapter: AT,
    latency: u64,
    initiation_interval: u64,
    device: tch::Device,
}

impl<T: Send + Sync + 'static, RT: DAMType, ST: DAMType, AT> Context
    for InferenceEngine<T, RT, ST, AT>
where
    AT: AdapterType<RT, ST, T> + Sync + Send,
    Model<T>: Job,
{
    fn run(&mut self) {
        loop {
            match self.communicator.recv() {
                Ok((time, batch)) => {
                    self.time.advance(time);

                    // Convert the batch into a TCH Input
                    let inputs = self.adapter.to_input(batch);

                    // run the model
                    let raw_outputs =
                        job_manager().with_license(self.device, &self.model, |jref| {
                            self.adapter.run(jref.job().as_ref().unwrap(), inputs)
                        });

                    let target_time = self.time.tick() + self.latency;
                    for output in self.adapter.from_output(raw_outputs) {
                        self.output
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: target_time,
                                    data: output,
                                },
                            )
                            .unwrap();
                    }
                    self.time.incr_cycles(self.initiation_interval);
                }

                // Our job here is finished.
                Err(_) => return,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, Mutex, RwLock};

    use dam::{
        context_tools::ChannelElement,
        simulation::ProgramBuilder,
        utility_contexts::{BroadcastContext, FunctionContext, GeneratorContext},
    };
    use tch::CModule;

    use crate::{manager::Job, model::Model, resources::add_ten_cmodule};

    use super::{AdapterType, ModelContext};

    struct BasicAdapter {}

    impl AdapterType<f64, f64, CModule> for BasicAdapter {
        fn to_input(&self, input: Vec<f64>) -> Vec<tch::Tensor> {
            vec![tch::Tensor::from_slice(&input)]
        }

        fn from_output(&self, model_output: tch::Tensor) -> Vec<f64> {
            model_output.iter::<f64>().unwrap().collect()
        }

        fn run(&self, model: &CModule, input: Vec<tch::Tensor>) -> tch::Tensor {
            model.forward_ts(&input).unwrap()
        }
    }

    #[test]
    fn basic_test() {
        const NUM_MODELS: usize = 32;
        const WAIT_LATENCY: u64 = 1024;
        const MAX_BATCH_SIZE: usize = 8;
        const MODEL_LATENCY: u64 = 73;
        const MODEL_II: u64 = 1;
        const NUM_INPUTS: usize = 64;
        let device = tch::Device::cuda_if_available();

        let mut ctx = ProgramBuilder::default();

        let (input_snd, input_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || {
                // Generate one f64 at a time.
                (0..NUM_INPUTS).map(|x| x as f64)
            },
            input_snd,
        ));

        let mut broadcaster = BroadcastContext::new(input_rcv);

        // stash the results into a list.
        let results = Arc::new(RwLock::new(vec![]));

        for _ in 0..NUM_MODELS {
            let (lsnd, lrcv) = ctx.unbounded();
            broadcaster.add_target(lsnd);
            let (output_snd, output_rcv) = ctx.unbounded();
            let mut model = Model::from_module(add_ten_cmodule());
            model.stash();
            ctx.add_child(ModelContext::new(
                lrcv,
                output_snd,
                Some(4),
                WAIT_LATENCY,
                MAX_BATCH_SIZE,
                model,
                BasicAdapter {},
                MODEL_LATENCY,
                MODEL_II,
                device,
            ));
            let stash = Arc::new(Mutex::new(vec![]));
            results.write().unwrap().push(stash.clone());
            let mut accumulator = FunctionContext::default();
            output_rcv.attach_receiver(&accumulator);
            accumulator.set_run(move |time| loop {
                match output_rcv.dequeue(&time) {
                    Ok(ce) => stash.lock().unwrap().push(ce),
                    Err(_) => return,
                }
            });
            ctx.add_child(accumulator);
        }

        ctx.add_child(broadcaster);
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());

        // At this point, results should have a list of timing traces
        let all_vecs = results.read().unwrap();
        for ind in 0..NUM_INPUTS {
            let mut values = vec![];
            for res in all_vecs.iter() {
                let value = res.lock().unwrap().get(ind).unwrap().clone();
                values.push(value);
            }
            let all_equals = values.windows(2).all(|slice| {
                let ChannelElement { time: t1, data: d1 } = slice[0];
                let ChannelElement { time: t2, data: d2 } = slice[1];
                t1 == t2 && d1 == d2
            });
            assert!(
                all_equals,
                "Mismatch between values at iteration {ind:?}: {:?}",
                values
            );
        }
    }
}
