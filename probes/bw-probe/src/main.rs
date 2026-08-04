// M0a feasibility probe: measure real H2D / D2H copy bandwidth on the
// dedicated transfer queue, plus expert-sized (8 MB) copy latency.
// This is the number the whole MoE-streaming design depends on.

use ash::vk;
use std::time::Instant;

const MB: u64 = 1024 * 1024;

struct Ctx {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    pdev: vk::PhysicalDevice,
    queue: vk::Queue,
    qfam: u32,
    qname: String,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl Ctx {
    unsafe fn new() -> Ctx {
        let entry = ash::Entry::load().expect("vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let ici = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = entry.create_instance(&ici, None).expect("instance");

        // pick the discrete GPU
        let pdevs = instance.enumerate_physical_devices().unwrap();
        let pdev = *pdevs
            .iter()
            .find(|p| {
                let props = instance.get_physical_device_properties(**p);
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .expect("no discrete GPU");
        let props = instance.get_physical_device_properties(pdev);
        let name = props
            .device_name_as_c_str()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!("device: {name}");

        // prefer a dedicated transfer queue family (TRANSFER but not GRAPHICS/COMPUTE)
        let qfams = instance.get_physical_device_queue_family_properties(pdev);
        let mut pick: Option<(u32, String)> = None;
        for (i, q) in qfams.iter().enumerate() {
            let f = q.queue_flags;
            let dedicated = f.contains(vk::QueueFlags::TRANSFER)
                && !f.contains(vk::QueueFlags::GRAPHICS)
                && !f.contains(vk::QueueFlags::COMPUTE);
            if dedicated {
                pick = Some((i as u32, format!("dedicated transfer (family {i}, flags {f:?})")));
                break;
            }
        }
        let (qfam, qname) = pick.unwrap_or_else(|| {
            let i = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    || q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .unwrap() as u32;
            (i, format!("fallback queue family {i} ({:?})", qfams[i as usize].queue_flags))
        });
        println!("queue:  {qname}");

        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&prio)];
        let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
        let device = instance.create_device(pdev, &dci, None).expect("device");
        let queue = device.get_device_queue(qfam, 0);
        let mem_props = instance.get_physical_device_memory_properties(pdev);

        Ctx { _entry: entry, instance, device, pdev, queue, qfam, qname, mem_props }
    }

    fn find_mem_type(&self, type_bits: u32, want: vk::MemoryPropertyFlags, avoid: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            let mt = self.mem_props.memory_types[i as usize];
            (type_bits & (1 << i)) != 0
                && mt.property_flags.contains(want)
                && !mt.property_flags.intersects(avoid)
        })
    }

    unsafe fn make_buffer(&self, size: u64, usage: vk::BufferUsageFlags, want: vk::MemoryPropertyFlags, avoid: vk::MemoryPropertyFlags) -> Option<(vk::Buffer, vk::DeviceMemory)> {
        let bci = vk::BufferCreateInfo::default().size(size).usage(usage);
        let buf = self.device.create_buffer(&bci, None).ok()?;
        let req = self.device.get_buffer_memory_requirements(buf);
        let mt = match self.find_mem_type(req.memory_type_bits, want, avoid) {
            Some(m) => m,
            None => {
                self.device.destroy_buffer(buf, None);
                return None;
            }
        };
        let mai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(mt);
        let mem = match self.device.allocate_memory(&mai, None) {
            Ok(m) => m,
            Err(_) => {
                self.device.destroy_buffer(buf, None);
                return None;
            }
        };
        self.device.bind_buffer_memory(buf, mem, 0).unwrap();
        Some((buf, mem))
    }
}

// time N copy submissions of `size` bytes, return GB/s and per-copy ms
unsafe fn bench_copy(ctx: &Ctx, src: vk::Buffer, dst: vk::Buffer, size: u64, iters: u32) -> (f64, f64) {
    let d = &ctx.device;
    let cpci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.qfam)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = d.create_command_pool(&cpci, None).unwrap();
    let cbai = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = d.allocate_command_buffers(&cbai).unwrap()[0];

    d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()).unwrap();
    let region = [vk::BufferCopy::default().size(size)];
    d.cmd_copy_buffer(cb, src, dst, &region);
    d.end_command_buffer(cb).unwrap();

    let fci = vk::FenceCreateInfo::default();
    let fence = d.create_fence(&fci, None).unwrap();
    let cbs = [cb];
    let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];

    // warmup
    for _ in 0..3 {
        d.queue_submit(ctx.queue, &submit, fence).unwrap();
        d.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        d.reset_fences(&[fence]).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        d.queue_submit(ctx.queue, &submit, fence).unwrap();
        d.wait_for_fences(&[fence], true, u64::MAX).unwrap();
        d.reset_fences(&[fence]).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    let gbps = (size as f64 * iters as f64) / dt / 1e9;
    let ms = dt / iters as f64 * 1e3;

    d.destroy_fence(fence, None);
    d.destroy_command_pool(pool, None);
    (gbps, ms)
}

fn main() {
    unsafe {
        let ctx = Ctx::new();

        // report memory heaps
        println!("\nmemory heaps:");
        for i in 0..ctx.mem_props.memory_heap_count {
            let h = ctx.mem_props.memory_heaps[i as usize];
            println!("  heap {i}: {:>6} MB  {:?}", h.size / MB, h.flags);
        }

        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let big = 256 * MB;

        // host-visible staging (GTT, cached if available)
        let host_cached = ctx.make_buffer(
            big, usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let host_uncached = ctx.make_buffer(
            big, usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_CACHED,
        );
        // device-local target
        let (dbuf, dmem) = ctx.make_buffer(
            big, usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        ).expect("device-local buffer");
        // ReBAR: device-local + host-visible
        let rebar = ctx.make_buffer(
            big, usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::empty(),
        );

        println!("\n== bulk copies, 256 MB x 10 iters, queue: {} ==", ctx.qname);
        if let Some((hbuf, _)) = host_cached {
            let (g, m) = bench_copy(&ctx, hbuf, dbuf, big, 10);
            println!("H2D (host-cached staging -> VRAM):   {g:6.2} GB/s  ({m:.2} ms / 256MB)");
            let (g2, m2) = bench_copy(&ctx, dbuf, hbuf, big, 10);
            println!("D2H (VRAM -> host-cached staging):   {g2:6.2} GB/s  ({m2:.2} ms / 256MB)");
        } else {
            println!("no host-cached staging type available");
        }
        if let Some((hbuf, _)) = host_uncached {
            let (g, m) = bench_copy(&ctx, hbuf, dbuf, big, 10);
            println!("H2D (uncached/WC staging -> VRAM):   {g:6.2} GB/s  ({m:.2} ms / 256MB)");
        }
        if let Some((rbuf, _)) = rebar {
            let (g, m) = bench_copy(&ctx, rbuf, dbuf, big, 10);
            println!("ReBAR(dev+host) -> VRAM copy:        {g:6.2} GB/s  ({m:.2} ms / 256MB)");
        } else {
            println!("no ReBAR (device-local + host-visible) memory type");
        }

        // expert-sized copies: 8 MB (typical Q4 expert slice) — latency matters
        println!("\n== expert-sized copies (submission latency included) ==");
        if let Some((hbuf, _)) = host_cached.or(host_uncached) {
            for sz_mb in [2u64, 8, 16, 32] {
                let sz = sz_mb * MB;
                let (g, m) = bench_copy(&ctx, hbuf, dbuf, sz, 50);
                println!("H2D {sz_mb:>3} MB: {g:6.2} GB/s effective, {:.3} ms/copy", m);
            }
        }

        ctx.device.destroy_device(None);
        ctx.instance.destroy_instance(None);
        let _ = ctx.pdev;
        let _ = dmem;
        println!("\ndone");
    }
}
