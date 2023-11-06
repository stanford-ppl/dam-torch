use std::sync::{atomic::AtomicUsize, Arc, Mutex};

use dashmap::DashMap;
use tch::Device;

pub trait Job {
    fn load(&mut self, device: Device);
    fn stash(&mut self);
}

#[derive(Clone)]
pub struct JobRef {
    id: ModelHandle,
    job: Arc<Mutex<dyn Job>>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ModelHandle(usize);

static NEXT_HANDLE: AtomicUsize = AtomicUsize::new(0);
impl ModelHandle {
    pub fn new() -> Self {
        ModelHandle(NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

struct DeviceManager {
    device: Device,
    resident: Mutex<Option<JobRef>>,
}

impl DeviceManager {
    /// with_license waits until the device is open, and then evicts the current job, replacing it with the new job.
    /// For the duration of the license, the device is locked, preventing other jobs from evicting it.
    pub fn with_license<F, T>(&self, jobref: JobRef, callback: F) -> T
    where
        F: FnOnce(JobRef) -> T,
    {
        let mut guard = self.resident.lock().unwrap();
        match guard.as_ref() {
            Some(current) if current.id == jobref.id => {
                // Already there, we don't need to do anything
            }
            Some(current) => {
                // stash the current job
                current.job.lock().unwrap().stash();

                // Load the new job onto the device.
                jobref.job.lock().unwrap().load(self.device);
                *guard = Some(jobref.clone());
            }
            None => {
                jobref.job.lock().unwrap().load(self.device);
                // Place the current job onto the device
                *guard = Some(jobref.clone());
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

pub struct JobManager {
    /// A map from devices to what's currently running.
    occupants: DashMap<Device, Arc<DeviceManager>>,
}

impl JobManager {
    pub fn with_license<F, T>(&self, device: Device, jobref: JobRef, callback: F) -> T
    where
        F: FnOnce(JobRef) -> T,
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
