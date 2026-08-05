// M7b kill probe: how fast can a COMPUTE SHADER move bytes GTT -> VRAM?
//
// Known brackets: DMA copy engine does 28 GB/s (bw-probe); the matvec
// shader reading experts in-place from GTT achieves ~5.8 GiB/s effective.
// If a coalesced gather lands near 28, staging cold slabs into VRAM
// scratch is a real lever. If it lands near 5.8, GTT reads are capped
// regardless of access pattern and M7b's staging design dies here.
//
// Modes (gather.comp):
//   0 = contiguous gather GTT -> VRAM (the design's core op)
//   1 = scattered GTT reads, checksum sink (access-pattern sensitivity)
//   2 = constant writes to VRAM (isolate VRAM write path)
//   3 = contiguous GTT reads, checksum sink (pure GTT read ceiling)

use ash::vk;
use std::time::Instant;

const MB: u64 = 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    slab_vecs: u32,
    n_slabs: u32,
    mode: u32,
    _pad: u32,
}
impl Push {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

unsafe fn find_mem_type(
    mp: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    want: vk::MemoryPropertyFlags,
    avoid: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mp.memory_type_count).find(|&i| {
        let mt = mp.memory_types[i as usize];
        (bits & (1 << i)) != 0
            && mt.property_flags.contains(want)
            && !mt.property_flags.intersects(avoid)
    })
}

fn main() {
    unsafe {
        let entry = ash::Entry::load().unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let ici = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = entry.create_instance(&ici, None).unwrap();
        let pdevs = instance.enumerate_physical_devices().unwrap();
        let pdev = *pdevs
            .iter()
            .find(|p| {
                instance.get_physical_device_properties(**p).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap();
        let name = instance
            .get_physical_device_properties(pdev)
            .device_name_as_c_str()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!("device: {name}");

        let qfams = instance.get_physical_device_queue_family_properties(pdev);
        let qfam = qfams
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&prio)];
        let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
        let device = instance.create_device(pdev, &dci, None).unwrap();
        let queue = device.get_device_queue(qfam, 0);
        let mem_props = instance.get_physical_device_memory_properties(pdev);

        // shader module from SPIR-V bytes
        let spv = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/gather.comp.spv"))
            .expect("compile gather.comp with glslc first");
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let smci = vk::ShaderModuleCreateInfo::default().code(&words);
        let sm = device.create_shader_module(&smci, None).unwrap();

        // descriptor set layout: 3 storage buffers (src, dst, sink)
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dslci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let dsl_layout = device.create_descriptor_set_layout(&dslci, None).unwrap();

        let layouts = [dsl_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let prci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_ranges);
        let pl = device.create_pipeline_layout(&prci, None).unwrap();

        let main_name = std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(sm)
            .name(main_name);
        let cpci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pl);
        let pipelines = [cpci];
        let pipe = device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipelines, None)
            .unwrap()[0];

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(32)];
        let dpci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(8)
            .pool_sizes(&pool_sizes);
        let dpool = device.create_descriptor_pool(&dpci, None).unwrap();

        let big = 64 * MB;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;

        let make_buffer =
            |size: u64, want: vk::MemoryPropertyFlags, avoid: vk::MemoryPropertyFlags| {
                let bci = vk::BufferCreateInfo::default().size(size).usage(usage);
                let buf = device.create_buffer(&bci, None).unwrap();
                let req = device.get_buffer_memory_requirements(buf);
                let mt = find_mem_type(&mem_props, req.memory_type_bits, want, avoid)
                    .expect("memory type");
                let mai = vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mt);
                let mem = device.allocate_memory(&mai, None).unwrap();
                device.bind_buffer_memory(buf, mem, 0).unwrap();
                buf
            };

        // GTT source (write-combine, the type expert tensors use)
        let gbuf = make_buffer(
            big,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_CACHED,
        );
        // VRAM destination
        let vbuf = make_buffer(
            big,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        let alloc_sets = [dsl_layout];
        let dai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(dpool)
            .set_layouts(&alloc_sets);
        let alloc = device.allocate_descriptor_sets(&dai).unwrap()[0];

        let gi = [vk::DescriptorBufferInfo::default()
            .buffer(gbuf)
            .offset(0)
            .range(big)];
        let vi = [vk::DescriptorBufferInfo::default()
            .buffer(vbuf)
            .offset(0)
            .range(big)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(alloc)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gi),
            vk::WriteDescriptorSet::default()
                .dst_set(alloc)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&vi),
            vk::WriteDescriptorSet::default()
                .dst_set(alloc)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&vi),
        ];
        device.update_descriptor_sets(&writes, &[]);

        let cpci2 = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qfam)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = device.create_command_pool(&cpci2, None).unwrap();
        let cbai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = device.allocate_command_buffers(&cbai).unwrap()[0];
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();

        let run = |mode: u32, bytes: u64, iters: u32| -> (f64, f64) {
            let slab = 1 * MB;
            let n_slabs = (bytes / slab) as u32;
            let slab_vecs = (slab / 16) as u32;
            let total_threads = n_slabs as u64 * slab_vecs as u64;
            let push = Push {
                slab_vecs,
                n_slabs,
                mode,
                _pad: 0,
            };
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cb, &begin_info).unwrap();
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe);
            let sets = [alloc];
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pl, 0, &sets, &[]);
            device.cmd_push_constants(cb, pl, vk::ShaderStageFlags::COMPUTE, 0, push.as_bytes());
            device.cmd_dispatch(cb, (total_threads as u32).div_ceil(256), 1, 1);
            device.end_command_buffer(cb).unwrap();
            let cbs = [cb];
            let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
            let fences = [fence];
            for _ in 0..3 {
                device.queue_submit(queue, &submit, fence).unwrap();
                device.wait_for_fences(&fences, true, u64::MAX).unwrap();
                device.reset_fences(&fences).unwrap();
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                device.queue_submit(queue, &submit, fence).unwrap();
                device.wait_for_fences(&fences, true, u64::MAX).unwrap();
                device.reset_fences(&fences).unwrap();
            }
            let dt = t0.elapsed().as_secs_f64();
            let gbps = (bytes as f64 * iters as f64) / dt / 1e9;
            (gbps, dt / iters as f64 * 1e3)
        };

        println!("\n== compute gather GTT -> VRAM (mode 0), 1 MB slabs ==");
        for mb in [8u64, 32, 64] {
            let (g, m) = run(0, mb * MB, 20);
            println!("  {mb:>3} MB gather:     {g:6.2} GB/s  ({m:.3} ms)");
        }
        println!("\n== scattered GTT reads (mode 1) ==");
        for mb in [8u64, 64] {
            let (g, m) = run(1, mb * MB, 20);
            println!("  {mb:>3} MB scatter:    {g:6.2} GB/s  ({m:.3} ms)");
        }
        println!("\n== contiguous GTT reads only (mode 3) ==");
        for mb in [8u64, 64] {
            let (g, m) = run(3, mb * MB, 20);
            println!("  {mb:>3} MB read-only:  {g:6.2} GB/s  ({m:.3} ms)");
        }
        println!("\n== VRAM writes only (mode 2) ==");
        let (g, m) = run(2, 64 * MB, 20);
        println!("   64 MB write-only:  {g:6.2} GB/s  ({m:.3} ms)");

        device.destroy_device(None);
        instance.destroy_instance(None);
        println!("\ndone");
    }
}
