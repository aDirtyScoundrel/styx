//! Minimal Vulkan compute backend for moe-stream.
//!
//! Wraps ash with just what the engine needs: device init on the discrete
//! GPU, host-visible + device-local buffer helpers, SPIR-V pipeline creation
//! with specialization constants, and synchronous dispatch. The pipeline ABI
//! mirrors ggml-vulkan: N storage-buffer bindings in one descriptor set plus
//! one push-constant range.

use ash::vk;
use std::path::Path;

pub mod ops;

pub struct Buffer {
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: u64,
    host_visible: bool,
}

pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub dset_layout: vk::DescriptorSetLayout,
    pub module: vk::ShaderModule,
    pub n_bindings: u32,
}

/// One command buffer + its own descriptor pool + fence. Record many
/// dispatches, then submit once. Not thread-safe; one per graph.
pub struct Batch {
    dpool: vk::DescriptorPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    recording: bool,
}

impl Batch {
    /// Must be called before the first dispatch in a new graph.
    pub fn begin(&mut self, gpu: &Gpu) -> Result<(), String> {
        assert!(!self.recording);
        unsafe {
            gpu.device
                .begin_command_buffer(self.cb, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| e.to_string())?;
        }
        self.recording = true;
        Ok(())
    }

    /// Record one dispatch. Buffers are bound sequentially to bindings 0..N.
    /// The descriptor set is allocated from the batch's private pool.
    pub fn dispatch(
        &mut self,
        gpu: &Gpu,
        p: &Pipeline,
        buffers: &[&Buffer],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<(), String> {
        let ranges: Vec<(&Buffer, u64, u64)> =
            buffers.iter().map(|b| (*b, 0, vk::WHOLE_SIZE)).collect();
        self.dispatch_ranges(gpu, p, &ranges, push, groups)
    }

    /// Like `dispatch`, but each binding is a (buffer, offset_bytes, range_bytes)
    /// sub-range (range = vk::WHOLE_SIZE for "rest of buffer"). Offsets must
    /// respect minStorageBufferOffsetAlignment (64 B on RADV).
    pub fn dispatch_ranges(
        &mut self,
        gpu: &Gpu,
        p: &Pipeline,
        buffers: &[(&Buffer, u64, u64)],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<(), String> {
        self.dispatch_ranges_barrier(gpu, p, buffers, push, groups, true)
    }

    /// `barrier=false` skips the trailing compute->compute barrier; use for
    /// dispatches whose outputs are not read by the IMMEDIATELY following
    /// dispatch (e.g. q/k/v or gate/up sibling matvecs).
    pub fn dispatch_ranges_barrier(
        &mut self,
        gpu: &Gpu,
        p: &Pipeline,
        buffers: &[(&Buffer, u64, u64)],
        push: &[u8],
        groups: (u32, u32, u32),
        barrier: bool,
    ) -> Result<(), String> {
        assert!(self.recording);
        assert_eq!(buffers.len() as u32, p.n_bindings);
        unsafe {
            let layouts = [p.dset_layout];
            let dset = gpu
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.dpool)
                        .set_layouts(&layouts),
                )
                .map_err(|e| e.to_string())?[0];

            let infos: Vec<_> = buffers
                .iter()
                .map(|(b, off, range)| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b.buf)
                        .offset(*off)
                        .range(*range)]
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(dset)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            gpu.device.update_descriptor_sets(&writes, &[]);

            gpu.device
                .cmd_bind_pipeline(self.cb, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            gpu.device.cmd_bind_descriptor_sets(
                self.cb,
                vk::PipelineBindPoint::COMPUTE,
                p.layout,
                0,
                &[dset],
                &[],
            );
            if !push.is_empty() {
                gpu.device.cmd_push_constants(
                    self.cb,
                    p.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            gpu.device
                .cmd_dispatch(self.cb, groups.0, groups.1, groups.2);
            // Barrier: ensure writes from this dispatch are visible to next.
            // Global memory barrier for all bound storage buffers.
            if barrier {
                gpu.device.cmd_pipeline_barrier(
                    self.cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)],
                    &[],
                    &[],
                );
            }
            Ok(())
        }
    }

    /// Submit the recorded command buffer once, wait on its fence.
    pub fn submit(&mut self, gpu: &Gpu) -> Result<(), String> {
        if !self.recording {
            return Ok(()); // no work
        }
        unsafe {
            gpu.device
                .end_command_buffer(self.cb)
                .map_err(|e| e.to_string())?;
            let cbs = [self.cb];
            gpu.device
                .queue_submit(
                    gpu.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cbs)],
                    self.fence,
                )
                .map_err(|e| e.to_string())?;
            gpu.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| e.to_string())?;
            gpu.device
                .reset_fences(&[self.fence])
                .map_err(|e| e.to_string())?;
            gpu.device
                .reset_command_buffer(self.cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| e.to_string())?;
            gpu.device
                .reset_descriptor_pool(self.dpool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| e.to_string())?;
            self.recording = false;
            Ok(())
        }
    }
}

pub struct Gpu {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub qfam: u32,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub device_name: String,
    cmd_pool: vk::CommandPool,
    dpool: vk::DescriptorPool,
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| e.to_string())?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let ici = vk::InstanceCreateInfo::default().application_info(&app);
            let instance = entry
                .create_instance(&ici, None)
                .map_err(|e| e.to_string())?;

            let pdevs = instance
                .enumerate_physical_devices()
                .map_err(|e| e.to_string())?;
            let pdev = *pdevs
                .iter()
                .find(|p| {
                    instance.get_physical_device_properties(**p).device_type
                        == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .or_else(|| pdevs.first())
                .ok_or("no Vulkan device")?;
            let props = instance.get_physical_device_properties(pdev);
            let device_name = props
                .device_name_as_c_str()
                .unwrap_or(c"?")
                .to_string_lossy()
                .into_owned();

            let qfams = instance.get_physical_device_queue_family_properties(pdev);
            let qfam = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("no compute queue")? as u32;

            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qfam)
                .queue_priorities(&prio)];

            // Features the ggml shaders require.
            let mut f16i8 = vk::PhysicalDeviceVulkan12Features::default()
                .shader_float16(true)
                .shader_int8(true)
                .storage_buffer8_bit_access(true)
                .uniform_and_storage_buffer8_bit_access(true);
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(true)
                .uniform_and_storage_buffer16_bit_access(true);
            let features = vk::PhysicalDeviceFeatures::default().shader_int16(true);
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .enabled_features(&features)
                .push_next(&mut f16i8)
                .push_next(&mut f11);
            let device = instance
                .create_device(pdev, &dci, None)
                .map_err(|e| e.to_string())?;
            let queue = device.get_device_queue(qfam, 0);
            let mem_props = instance.get_physical_device_memory_properties(pdev);

            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qfam)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| e.to_string())?;

            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(256)];
            let dpool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(64)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(|e| e.to_string())?;

            Ok(Gpu {
                _entry: entry,
                instance,
                pdev,
                device,
                queue,
                qfam,
                mem_props,
                device_name,
                cmd_pool,
                dpool,
            })
        }
    }

    fn find_mem_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            (bits & (1 << i)) != 0
                && self.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(want)
        })
    }

    fn find_mem_type_not(
        &self,
        bits: u32,
        want: vk::MemoryPropertyFlags,
        avoid: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            let f = self.mem_props.memory_types[i as usize].property_flags;
            (bits & (1 << i)) != 0 && f.contains(want) && !f.intersects(avoid)
        })
    }

    pub fn create_buffer(&self, size: u64, host_visible: bool) -> Result<Buffer, String> {
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            let buf = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buf);
            // Prefer ReBAR (DEVICE_LOCAL + HOST_VISIBLE): mappable AND full
            // VRAM bandwidth. Plain HOST_VISIBLE lands in GTT (PCIe-limited).
            let mt = if host_visible {
                self.find_mem_type(
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL
                        | vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .or_else(|| {
                    self.find_mem_type(
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                })
            } else {
                self.find_mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            }
            .ok_or("no matching memory type")?;
            let mem = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device
                .bind_buffer_memory(buf, mem, 0)
                .map_err(|e| e.to_string())?;
            Ok(Buffer {
                buf,
                mem,
                size,
                host_visible,
            })
        }
    }

    /// Host-memory (GTT) buffer: HOST_VISIBLE but explicitly NOT device-local.
    /// The GPU reads it over PCIe. For oversize weights (expert streaming).
    pub fn create_buffer_host(&self, size: u64) -> Result<Buffer, String> {
        unsafe {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST;
            let buf = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let mt = self
                .find_mem_type_not(
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or("no host-only memory type")?;
            let mem = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(mt),
                    None,
                )
                .map_err(|e| e.to_string())?;
            self.device
                .bind_buffer_memory(buf, mem, 0)
                .map_err(|e| e.to_string())?;
            Ok(Buffer {
                buf,
                mem,
                size,
                host_visible: true,
            })
        }
    }

    pub fn upload(&self, b: &Buffer, data: &[u8]) -> Result<(), String> {
        assert!(b.host_visible, "upload targets host-visible buffers");
        assert!(data.len() as u64 <= b.size);
        unsafe {
            let ptr = self
                .device
                .map_memory(b.mem, 0, data.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| e.to_string())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            self.device.unmap_memory(b.mem);
        }
        Ok(())
    }

    pub fn download(&self, b: &Buffer, out: &mut [u8]) -> Result<(), String> {
        assert!(b.host_visible);
        assert!(out.len() as u64 <= b.size);
        unsafe {
            let ptr = self
                .device
                .map_memory(b.mem, 0, out.len() as u64, vk::MemoryMapFlags::empty())
                .map_err(|e| e.to_string())?;
            std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), out.len());
            self.device.unmap_memory(b.mem);
        }
        Ok(())
    }

    /// Load a SPIR-V compute pipeline with `n_bindings` storage buffers,
    /// a push-constant range of `push_bytes`, and u32 specialization
    /// constants `(constant_id, value)`.
    pub fn create_pipeline(
        &self,
        spv_path: &Path,
        n_bindings: u32,
        push_bytes: u32,
        spec: &[(u32, u32)],
    ) -> Result<Pipeline, String> {
        let bytes = std::fs::read(spv_path).map_err(|e| format!("{spv_path:?}: {e}"))?;
        if bytes.len() % 4 != 0 {
            return Err("SPIR-V size not multiple of 4".into());
        }
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        unsafe {
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(|e| e.to_string())?;

            let bindings: Vec<_> = (0..n_bindings)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dset_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| e.to_string())?;

            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_bytes)];
            let layouts = [dset_layout];
            let layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .map_err(|e| e.to_string())?;

            let entries: Vec<_> = spec
                .iter()
                .enumerate()
                .map(|(i, (id, _))| {
                    vk::SpecializationMapEntry::default()
                        .constant_id(*id)
                        .offset(i as u32 * 4)
                        .size(4)
                })
                .collect();
            let spec_data: Vec<u8> = spec.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&spec_data);

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main")
                .specialization_info(&spec_info);
            let ci = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)];
            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &ci, None)
                .map_err(|(_, e)| e.to_string())?[0];

            Ok(Pipeline {
                pipeline,
                layout,
                dset_layout,
                module,
                n_bindings,
            })
        }
    }

    /// Bind buffers to bindings 0..N in order, push constants, dispatch, wait.
    pub fn dispatch_sync(
        &self,
        p: &Pipeline,
        buffers: &[&Buffer],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<(), String> {
        assert_eq!(buffers.len() as u32, p.n_bindings);
        unsafe {
            let layouts = [p.dset_layout];
            let dset = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.dpool)
                        .set_layouts(&layouts),
                )
                .map_err(|e| e.to_string())?[0];

            let infos: Vec<_> = buffers
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b.buf)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(dset)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);

            let cb = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| e.to_string())?[0];
            self.device
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())
                .map_err(|e| e.to_string())?;
            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                p.layout,
                0,
                &[dset],
                &[],
            );
            if !push.is_empty() {
                self.device.cmd_push_constants(
                    cb,
                    p.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            self.device.cmd_dispatch(cb, groups.0, groups.1, groups.2);
            self.device
                .end_command_buffer(cb)
                .map_err(|e| e.to_string())?;

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| e.to_string())?;
            let cbs = [cb];
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&cbs)],
                    fence,
                )
                .map_err(|e| e.to_string())?;
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| e.to_string())?;
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.cmd_pool, &[cb]);
            // Synchronous dispatch: the set is no longer in use — recycle the
            // pool so repeated dispatches can't exhaust it.
            self.device
                .reset_descriptor_pool(self.dpool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| e.to_string())?;
            Ok(())
        }
    }

    pub fn destroy_buffer(&self, b: Buffer) {
        unsafe {
            self.device.destroy_buffer(b.buf, None);
            self.device.free_memory(b.mem, None);
        }
    }

    /// Create a reusable batch recorder: many dispatches, one submit/fence.
    /// `max_sets`/`max_descriptors` size its private descriptor pool — pick
    /// them to cover one full recorded graph (e.g. a whole token).
    pub fn create_batch(&self, max_sets: u32, max_descriptors: u32) -> Result<Batch, String> {
        unsafe {
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(max_descriptors)];
            let dpool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(max_sets)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(|e| e.to_string())?;
            let cb = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| e.to_string())?[0];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| e.to_string())?;
            Ok(Batch {
                dpool,
                cb,
                fence,
                recording: false,
            })
        }
    }

    pub fn destroy_batch(&self, b: Batch) {
        unsafe {
            self.device.destroy_fence(b.fence, None);
            self.device.free_command_buffers(self.cmd_pool, &[b.cb]);
            self.device.destroy_descriptor_pool(b.dpool, None);
        }
    }

    pub fn destroy_pipeline(&self, p: Pipeline) {
        unsafe {
            self.device.destroy_pipeline(p.pipeline, None);
            self.device.destroy_pipeline_layout(p.layout, None);
            self.device
                .destroy_descriptor_set_layout(p.dset_layout, None);
            self.device.destroy_shader_module(p.module, None);
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_descriptor_pool(self.dpool, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
