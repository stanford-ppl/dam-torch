use std::sync::{atomic::AtomicUsize, Arc, Mutex, MutexGuard, OnceLock, Weak};

use dashmap::DashMap;
use tch::Device;

pub trait Job: Sync + Send {
    fn load(&mut self, device: Device);
    fn stash(&mut self);
}

pub struct JobRef<T> {
    id: ModelHandle,
    job: Arc<Mutex<T>>,
}

impl<T> Clone for JobRef<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            job: self.job.clone(),
        }
    }
}

impl<T> JobRef<T>
where
    T: Job,
{
    pub fn new(job: T) -> Self {
        Self {
            id: ModelHandle::new(),
            job: Arc::new(Mutex::new(job)),
        }
    }

    pub fn job<'a>(&'a self) -> MutexGuard<'a, T> {
        self.job.lock().unwrap()
    }

    fn downgrade(&self) -> impl JobView {
        JobProxy::<T> {
            id: self.id,
            job: Arc::downgrade(&self.job),
        }
    }
}

struct JobProxy<T> {
    id: ModelHandle,
    job: Weak<Mutex<T>>,
}

impl<T> JobView for JobProxy<T>
where
    T: Send + Job,
{
    fn id(&self) -> ModelHandle {
        self.id
    }

    fn stash(&self) {
        if let Some(cur) = self.job.upgrade() {
            cur.lock().unwrap().stash();
        }
    }
}

impl<T> std::fmt::Debug for JobProxy<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobProxy")
            .field("id", &self.id)
            .field("job", &self.job)
            .finish()
    }
}

pub trait JobView: Sync + Send + std::fmt::Debug {
    fn id(&self) -> ModelHandle;
    fn stash(&self);
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ModelHandle(usize);

static NEXT_HANDLE: AtomicUsize = AtomicUsize::new(0);
impl ModelHandle {
    pub fn new() -> Self {
        ModelHandle(NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

#[derive(Debug)]
struct DeviceManager {
    device: Device,
    resident: Mutex<Option<Box<dyn JobView>>>,
}

impl DeviceManager {
    /// with_license waits until the device is open, and then evicts the current job, replacing it with the new job.
    /// For the duration of the license, the device is locked, preventing other jobs from evicting it.
    pub fn with_license<JT: Job + 'static, F, T>(&self, jobref: &JobRef<JT>, callback: F) -> T
    where
        F: FnOnce(&JobRef<JT>) -> T,
    {
        let mut guard = self.resident.lock().unwrap();
        match guard.as_ref() {
            Some(current) if current.id() == jobref.id => {
                // Already there, we don't need to do anything
            }
            Some(current) => {
                // stash the current job, if it hasn't been dropped yet.
                current.stash();

                // Load the new job onto the device.
                jobref.job.lock().unwrap().load(self.device);
                *guard = Some(Box::new(jobref.downgrade()));
            }
            None => {
                jobref.job.lock().unwrap().load(self.device);
                // Place the current job onto the device
                *guard = Some(Box::new(jobref.downgrade()));
            }
        }
        let result = callback(jobref);

        // Explicitly do this at the end to unlock the device. Otherwise it might be dropped earlier.
        drop(guard);

        result
    }
}

// A Manageable is a job that can be "moved" around
// In order to run, it needs to first acquire an exclusive license on a device.
// Once it's on a device, it wants to stay, but can be evicted if it's not currently running.

#[derive(Debug, Default)]
pub struct JobManager {
    /// A map from devices to what's currently running.
    occupants: DashMap<Device, Arc<DeviceManager>>,
}

impl JobManager {
    pub fn with_license<JT: Job + 'static, F, T>(
        &self,
        device: Device,
        jobref: &JobRef<JT>,
        callback: F,
    ) -> T
    where
        F: FnOnce(&JobRef<JT>) -> T,
    {
        let manager = self
            .occupants
            .entry(device)
            .or_insert_with(|| {
                Arc::new(DeviceManager {
                    device,
                    resident: Default::default(),
                })
            })
            .value()
            .clone();

        manager.with_license(jobref, callback)
    }
}

static MANAGER: OnceLock<JobManager> = OnceLock::new();
pub fn job_manager() -> &'static JobManager {
    MANAGER.get_or_init(Default::default)
}

#[cfg(test)]
mod test {
    use crate::{
        manager::{job_manager, Job, JobRef},
        model::Model,
        resources::add_ten_cmodule,
    };

    #[test]
    fn test_management() {
        let mut model = Model::from_module(add_ten_cmodule());
        let mut model2 = Model::from_module(add_ten_cmodule());
        println!("Models loaded");
        model.stash();
        model2.stash();

        let ref1 = JobRef::new(model);
        let ref2 = JobRef::new(model2);

        let test_tensor = tch::Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4]);

        let device = tch::Device::cuda_if_available();

        let r1 = job_manager().with_license(device, &ref1, |jref| {
            jref.job()
                .as_ref()
                .unwrap()
                .forward_ts(&[&test_tensor])
                .unwrap()
        });
        let r2 = job_manager().with_license(device, &ref2, |jref| {
            jref.job()
                .as_ref()
                .unwrap()
                .forward_ts(&[&test_tensor])
                .unwrap()
        });

        dbg!(device);
        let _ = dbg!(r1);
        let _ = dbg!(r2);
    }

    #[test]
    fn test_management_parallel() {
        let mut model = Model::from_module(add_ten_cmodule());
        let mut model2 = Model::from_module(add_ten_cmodule());
        println!("Models loaded");
        model.stash();
        model2.stash();

        let ref1 = JobRef::new(model);
        let ref2 = JobRef::new(model2);

        let device = tch::Device::cuda_if_available();

        std::thread::scope(|scope| {
            let spawn = |job: JobRef<Model<tch::CModule>>| {
                scope.spawn(move || {
                    for i in 0..64 {
                        let test_tensor = tch::Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4]);
                        let ret = job_manager().with_license(device, &job, |jref| {
                            println!("Spawning Job, iteration {i:?}");
                            jref.job()
                                .as_ref()
                                .unwrap()
                                .forward_ts(&[&test_tensor])
                                .unwrap()
                        });

                        let _ = dbg!(ret);
                    }
                });
            };

            spawn(ref1);
            spawn(ref2);
        });
    }
}
