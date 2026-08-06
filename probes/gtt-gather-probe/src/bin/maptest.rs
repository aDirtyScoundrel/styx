use ash::vk;
fn main() {
    unsafe {
        let entry = ash::Entry::load().unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .unwrap();
        let pdevs = instance.enumerate_physical_devices().unwrap();
        let pdev = *pdevs
            .iter()
            .find(|p| {
                instance.get_physical_device_properties(**p).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .unwrap();
        let qfams = instance.get_physical_device_queue_family_properties(pdev);
        let qfam = qfams
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&prio)];
        let device = instance
            .create_device(
                pdev,
                &vk::DeviceCreateInfo::default().queue_create_infos(&qci),
                None,
            )
            .unwrap();
        let mp = instance.get_physical_device_memory_properties(pdev);
        println!("memory types:");
        for i in 0..mp.memory_type_count {
            let t = mp.memory_types[i as usize];
            println!("  {i}: heap{} {:?}", t.heap_index, t.property_flags);
        }
        for size_mb in [4u64, 16, 64] {
            let size = size_mb * 1024 * 1024;
            for concurrent in [false, true] {
                let fams = [qfam, qfam.min(mp.memory_type_count.saturating_sub(1))];
                let bci = vk::BufferCreateInfo::default().size(size).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                );
                let bci = if concurrent {
                    bci
                        .sharing_mode(vk::SharingMode::CONCURRENT)
                        .queue_family_indices(&fams[..1])
                } else {
                    bci
                };
                let buf = device.create_buffer(&bci, None).unwrap();
                let req = device.get_buffer_memory_requirements(buf);
                let mt = (0..mp.memory_type_count)
                    .find(|&i| {
                        (req.memory_type_bits & (1 << i)) != 0
                            && mp.memory_types[i as usize]
                                .property_flags
                                .contains(
                                    vk::MemoryPropertyFlags::HOST_VISIBLE
                                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                                )
                            && !mp.memory_types[i as usize]
                                .property_flags
                                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    })
                    .unwrap();
                let mem = device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(mt),
                        None,
                    )
                    .unwrap();
                device.bind_buffer_memory(buf, mem, 0).unwrap();
                let mode = if concurrent { "CONCURRENT" } else { "EXCLUSIVE " };
                match device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()) {
                    Ok(_) => println!("{size_mb} MB {mode} type{mt}: MAP OK"),
                    Err(e) => println!("{size_mb} MB {mode} type{mt}: MAP FAILED {e:?}"),
                }
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
        }
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
