// M7b-B kill probe: does a second queue family's GTT->VRAM gather run
// CONCURRENTLY with a busy compute kernel on the main queue?
//
// M0 found family 1 = 4x COMPUTE|TRANSFER queues. M7b-B's payoff depends
// on the gather overlapping layer compute. If RADV serializes the two
// queues, M7b-B dies before implementation.
//
// Method:
//   T_busy   = busy kernel alone on queue 0
//   T_gather = gather alone on queue 1
//   T_both   = both submitted concurrently (semaphores, one fence)
//   gather_during_busy = gather's own duration inside T_both (timestamps)
// Verdict: overlap if T_both ~ max(T_busy, T_gather); serialized if
// T_both ~ T_busy + T_gather.

use ash::vk;
use std::time::Instant;

const MB: u64 = 1024 * 1024;

#[repr(C)]
#[derive(Clone, Copy)]
struct GatherPush {
    vg: u32,
    vu: u32,
    vd: u32,
    base_g: u32,
    base_up: u32,
    base_d: u32,
    _pad: u32,
}
impl GatherPush {
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>()) }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
struct BusyPush {
    iters: u32,
    n: u32,
}
impl BusyPush {
    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>()) }
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
        for (i, q) in qfams.iter().enumerate() {
            println!("  family {i}: {:?} x{}", q.queue_flags, q.queue_count);
        }
        let fam0 = qfams
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;
        // second family: compute-capable and different from fam0
        let fam1 = qfams
            .iter()
            .enumerate()
            .find(|(i, q)| *i as u32 != fam0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32);
        let fam1 = match fam1 {
            Some(f) => f,
            None => {
                println!("NO second compute family — M7b-B async overlap impossible. KILL.");
                return;
            }
        };
        println!("main compute family: {fam0}, gather family: {fam1}");

        let prio = [1.0f32];
        let qci0 = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(fam0)
            .queue_priorities(&prio);
        let qci1 = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(fam1)
            .queue_priorities(&prio);
        let qcis = [qci0, qci1];
        let feats = vk::PhysicalDeviceFeatures::default();
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qcis)
            .enabled_features(&feats);
        let device = instance.create_device(pdev, &dci, None).unwrap();
        let q0 = device.get_device_queue(fam0, 0);
        let q1 = device.get_device_queue(fam1, 0);
        let mem_props = instance.get_physical_device_memory_properties(pdev);

        // shaders
        let load = |p: &str| -> Vec<u32> {
            let b = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/")
                .to_owned() + p)
                .expect("spv");
            b.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let make_module = |words: &[u32]| {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(words),
                    None,
                )
                .unwrap()
        };
        // NOTE: the probe uses the REAL M7b-A gather shader (7 bindings:
        // gate/up/down src, arena_g/u/d dst, ids) — its ABI matches
        // GatherPush and the descriptor writes below. The old
        // gather.comp (mode-based, 3 bindings) silently fell into its
        // checksum branch and copied nothing.
        let g_words = load("../../shaders/gather_slabs_f32.spv");
        let b_words = load("busy.comp.spv");
        let sm_g = make_module(&g_words);
        let sm_b = make_module(&b_words);

        // CONCURRENT sharing between the two families (probe simplicity)
        let fams = [fam0, fam1];
        let make_buf = |size: u64,
                        want: vk::MemoryPropertyFlags,
                        avoid: vk::MemoryPropertyFlags|
         -> (vk::Buffer, vk::DeviceMemory) {
            let bci = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&fams);
            let buf = device.create_buffer(&bci, None).unwrap();
            let req = device.get_buffer_memory_requirements(buf);
            let mt = find_mem_type(&mem_props, req.memory_type_bits, want, avoid)
                .expect("mem type");
            let mai = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(mt);
            let mem = device.allocate_memory(&mai, None).unwrap();
            device.bind_buffer_memory(buf, mem, 0).unwrap();
            (buf, mem)
        };

        // gather buffers: 64 MB GTT src -> 64 MB VRAM dst
        let big = 64 * MB;
        let (gsrc, _gm) = make_buf(
            big,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_CACHED,
        );
        // destination: ReBAR (device-local + host-visible) so the
        // correctness check can map it back and read the gathered data
        let (gdst, _dm) = make_buf(
            big,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::empty(),
        );
        // busy buffers: 32 MB VRAM, iterated
        let (bsrc, _bm) = make_buf(
            32 * MB,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );
        // gather ids dummy (all slots active): 8 ids
        let (ids, imem) = make_buf(
            64,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // descriptor layouts
        let sb = vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let mk_dsl = |n: u32| {
            let bindings: Vec<_> = (0..n).map(|b| sb.binding(b)).collect();
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .unwrap()
        };
        let dsl_g = mk_dsl(7); // gather: gate,up,down,arenaG,arenaU,arenaD,ids
        let dsl_b = mk_dsl(1); // busy: data

        let main_name = std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0");
        let mk_pipe = |sm: vk::ShaderModule, dsl: vk::DescriptorSetLayout, push: u32| {
            let layouts = [dsl];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push)];
            let pl = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&ranges),
                    None,
                )
                .unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(sm)
                .name(main_name);
            let pipes = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(pl)];
            let pipe = device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipes, None)
                .unwrap()[0];
            (pl, pipe)
        };
        let (pl_g, pipe_g) = mk_pipe(sm_g, dsl_g, 28);
        let (pl_b, pipe_b) = mk_pipe(sm_b, dsl_b, 8);

        // descriptor pools + sets
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(64)];
        let dpool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(16)
                    .pool_sizes(&pool_sizes),
                None,
            )
            .unwrap();
        let mk_set = |dsl: vk::DescriptorSetLayout| {
            let layouts = [dsl];
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(dpool)
                        .set_layouts(&layouts),
                )
                .unwrap()[0]
        };
        let set_g = mk_set(dsl_g);
        let set_b = mk_set(dsl_b);

        // gather uses one tensor as gate/up/down and one arena for all three
        let gi = [vk::DescriptorBufferInfo::default().buffer(gsrc).offset(0).range(big)];
        let di = [vk::DescriptorBufferInfo::default().buffer(gdst).offset(0).range(big)];
        let ii = [vk::DescriptorBufferInfo::default().buffer(ids).offset(0).range(64)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set_g).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&gi),
            vk::WriteDescriptorSet::default().dst_set(set_g).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&gi),
            vk::WriteDescriptorSet::default().dst_set(set_g).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&gi),
            vk::WriteDescriptorSet::default().dst_set(set_g).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&di),
            vk::WriteDescriptorSet::default().dst_set(set_g).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&di),
            vk::WriteDescriptorSet::default().dst_set(set_g).dst_binding(5).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&di),
            vk::WriteDescriptorSet::default().dst_set(set_g).dst_binding(6).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ii),
        ];
        device.update_descriptor_sets(&writes, &[]);
        let bi = [vk::DescriptorBufferInfo::default().buffer(bsrc).offset(0).range(32 * MB)];
        let w_b = [vk::WriteDescriptorSet::default()
            .dst_set(set_b)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&bi)];
        device.update_descriptor_sets(&w_b, &[]);

        // write ids: 8 active experts (ids 0..7)
        let ids_map = device
            .map_memory(imem, 0, 64, vk::MemoryMapFlags::empty())
            .unwrap() as *mut i32;
        for i in 0..8 {
            *ids_map.add(i) = i as i32;
        }
        device.unmap_memory(imem);

        // command pools per family
        let mk_pool = |fam: u32| {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(fam)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap()
        };
        let pool0 = mk_pool(fam0);
        let pool1 = mk_pool(fam1);
        let mk_cb = |pool: vk::CommandPool| {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0]
        };
        let cb_busy = mk_cb(pool0);
        let cb_gather = mk_cb(pool1);

        // gather command: 8 workgroups, 1 MB slab per tensor (16384 uvec4)
        device
            .begin_command_buffer(cb_gather, &vk::CommandBufferBeginInfo::default())
            .unwrap();
        device.cmd_bind_pipeline(cb_gather, vk::PipelineBindPoint::COMPUTE, pipe_g);
        let sets_g = [set_g];
        device.cmd_bind_descriptor_sets(cb_gather, vk::PipelineBindPoint::COMPUTE, pl_g, 0, &sets_g, &[]);
        let gp = GatherPush {
            vg: 131072,
            vu: 131072,
            vd: 131072,
            base_g: 0,
            base_up: 0,
            base_d: 0,
            _pad: 0,
        };
        device.cmd_push_constants(cb_gather, pl_g, vk::ShaderStageFlags::COMPUTE, 0, gp.as_bytes());
        device.cmd_dispatch(cb_gather, 8, 1, 1);
        device.end_command_buffer(cb_gather).unwrap();

        // busy command: tune iters so busy ~ 20 ms
        let n_elems = (32 * MB / 4) as u32;
        let mk_busy = |iters: u32| {
            device.reset_command_buffer(cb_busy, vk::CommandBufferResetFlags::empty()).unwrap();
            device
                .begin_command_buffer(cb_busy, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            device.cmd_bind_pipeline(cb_busy, vk::PipelineBindPoint::COMPUTE, pipe_b);
            let sets_b = [set_b];
            device.cmd_bind_descriptor_sets(cb_busy, vk::PipelineBindPoint::COMPUTE, pl_b, 0, &sets_b, &[]);
            let bp = BusyPush { iters, n: n_elems };
            device.cmd_push_constants(cb_busy, pl_b, vk::ShaderStageFlags::COMPUTE, 0, bp.as_bytes());
            device.cmd_dispatch(cb_busy, n_elems.div_ceil(256), 1, 1);
            device.end_command_buffer(cb_busy).unwrap();
        };

        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
        let fences = [fence];
        let cbs_busy = [cb_busy];
        let cbs_gather = [cb_gather];
        let submit_busy = [vk::SubmitInfo::default().command_buffers(&cbs_busy)];
        let submit_gather = [vk::SubmitInfo::default().command_buffers(&cbs_gather)];

        // calibrate busy to ~20 ms, WITH warmup so GPU clocks are boosted
        // before the baseline is taken (clock ramping inflated cold runs)
        let mut iters = 2000u32;
        let mut busy_ms;
        loop {
            mk_busy(iters);
            // warmup: 3 sustained submits at this iter count
            for _ in 0..3 {
                device.queue_submit(q0, &submit_busy, fence).unwrap();
                device.wait_for_fences(&fences, true, u64::MAX).unwrap();
                device.reset_fences(&fences).unwrap();
            }
            device.queue_submit(q0, &submit_busy, fence).unwrap();
            let t0 = Instant::now();
            device.wait_for_fences(&fences, true, u64::MAX).unwrap();
            busy_ms = t0.elapsed().as_secs_f64() * 1e3;
            device.reset_fences(&fences).unwrap();
            if busy_ms > 10.0 || iters > 1_000_000 {
                break;
            }
            iters *= 2;
        }
        println!("busy kernel: {iters} iters -> {busy_ms:.1} ms (warmed)");

        // CORRECTNESS: fill gsrc with a pattern, run gather, verify gdst.
        // (guards against the gather silently no-op'ing)
        {
            let src_map = device
                .map_memory(_gm, 0, big, vk::MemoryMapFlags::empty())
                .unwrap() as *mut u32;
            for i in 0..(big / 4) as usize {
                *src_map.add(i) = (i % 4096) as u32;
            }
            device.unmap_memory(_gm);
            device.queue_submit(q1, &submit_gather, fence).unwrap();
            device.wait_for_fences(&fences, true, u64::MAX).unwrap();
            device.reset_fences(&fences).unwrap();
            let dst_map = device
                .map_memory(_dm, 0, big, vk::MemoryMapFlags::empty())
                .unwrap() as *mut u32;
            // slot 0 arena_g region: uvec4 index 0..vg; check first words
            let mut ok = true;
            for i in 0..64usize {
                let got = *dst_map.add(i * 4);
                if got != (i * 4 % 4096) as u32 {
                    ok = false;
                    println!("  mismatch at word {i}: got {got}");
                    break;
                }
            }
            device.unmap_memory(_dm);
            if !ok {
                println!("GATHER CORRECTNESS FAILED — probe invalid");
                return;
            }
            println!("gather correctness: slot 0 arena_g matches source pattern");
        }

        // gather alone on q1 (warmed too)
        for _ in 0..3 {
            device.queue_submit(q1, &submit_gather, fence).unwrap();
            device.wait_for_fences(&fences, true, u64::MAX).unwrap();
            device.reset_fences(&fences).unwrap();
        }
        device.queue_submit(q1, &submit_gather, fence).unwrap();
        let t0 = Instant::now();
        device.wait_for_fences(&fences, true, u64::MAX).unwrap();
        let gather_ms = t0.elapsed().as_secs_f64() * 1e3;
        device.reset_fences(&fences).unwrap();
        println!("gather alone (q1, 48 MB): {gather_ms:.2} ms");

        // concurrent: busy on q0 + gather on q1, gather submitted slightly
        // after busy starts; total wall time measured on the busy fence.
        let mut total = 0.0f64;
        let mut gather_wall = 0.0f64;
        let runs = 5;
        let sem = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        for _ in 0..runs {
            mk_busy(iters);
            // submit gather first waiting on nothing; busy second. Then wait.
            // Use a binary semaphore: q0 busy signals; we just fence q0.
            // Measure wall for q0 fence (busy), and separately fence the gather.
            let fence_g = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();
            device.queue_submit(q0, &submit_busy, fence).unwrap();
            let t_g0 = Instant::now();
            device.queue_submit(q1, &submit_gather, fence_g).unwrap();
            let t0 = Instant::now();
            device.wait_for_fences(&fences, true, u64::MAX).unwrap();
            total += t0.elapsed().as_secs_f64() * 1e3;
            device.wait_for_fences(&[fence_g], true, u64::MAX).unwrap();
            gather_wall += t_g0.elapsed().as_secs_f64() * 1e3;
            device.reset_fences(&fences).unwrap();
            device.destroy_fence(fence_g, None);
            let _ = sem;
        }
        let avg_total = total / runs as f64;
        let avg_gather = gather_wall / runs as f64;
        println!("concurrent: busy(q0)+gather(q1) avg wall(busy fence) {avg_total:.2} ms; gather wall {avg_gather:.2} ms");

        let serial_expect = busy_ms + gather_ms;
        let overlap_expect = busy_ms.max(gather_ms);
        println!("expected if serialized: ~{serial_expect:.1} ms | if overlapped: ~{overlap_expect:.1} ms");
        if avg_total < busy_ms + gather_ms * 0.5 {
            println!("VERDICT: OVERLAP — M7b-B async gather is viable");
        } else {
            println!("VERDICT: SERIALIZED — M7b-B needs a different mechanism (or dies)");
        }

        device.destroy_device(None);
        instance.destroy_instance(None);
        println!("done");
    }
}
