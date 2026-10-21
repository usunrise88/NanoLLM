// vkbw: how fast can the Iris Xe read memory through a compute shader? A 1 GiB buffer in host-visible
// device-local memory (what llama.cpp's Vulkan backend uses on a UMA device) is summed by workgroups that each
// read a contiguous chunk, for several workgroup sizes, chunk sizes and load widths. The best number is the
// ceiling the mat-vec kernels (decode) can hope for.
//
//   glslc -O -fshader-stage=compute read.comp -o read.spv   (with -DVEC=1|2|4 and WG as spec constants)
//   vkbw read.spv
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static double now(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart; }

int main(int argc, char **argv) {
    const char *spv_path = argc > 1 ? argv[1] : "read.spv";
    FILE *f = fopen(spv_path, "rb");
    if (!f) { printf("no %s\n", spv_path); return 1; }
    fseek(f, 0, SEEK_END); long spv_size = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(spv_size); fread(spv, 1, spv_size, f); fclose(f);

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "vkbw", 1, "none", 1, VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &app };
    VkInstance inst; CHECK(vkCreateInstance(&ici, 0, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, 0);
    VkPhysicalDevice devs[8]; if (n > 8) n = 8; vkEnumeratePhysicalDevices(inst, &n, devs);
    VkPhysicalDevice pd = 0;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
        printf("device %u: %s (vendor %#x, type %d)\n", i, p.deviceName, p.vendorID, p.deviceType);
        if (p.vendorID == 0x8086 && !pd) pd = devs[i];
    }
    if (!pd) { printf("no Intel device\n"); return 1; }

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, 0);
    VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    uint32_t qf = 0; for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, qf, 1, &prio };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &dqi };
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, 0, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qf, 0, &q);

    const VkDeviceSize SIZE = 1ull << 30;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE };
    VkBuffer A, R; CHECK(vkCreateBuffer(dev, &bci, 0, &A));
    bci.size = 64 << 20; CHECK(vkCreateBuffer(dev, &bci, 0, &R));
    VkMemoryRequirements mrA, mrR; vkGetBufferMemoryRequirements(dev, A, &mrA); vkGetBufferMemoryRequirements(dev, R, &mrR);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    int mt = -1;
    VkMemoryPropertyFlags want[] = { VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT };
    for (int w = 0; w < 2 && mt < 0; w++)
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((mrA.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want[w]) == want[w]) { mt = (int)i; break; }
    printf("memory type %d flags %#x\n", mt, mp.memoryTypes[mt].propertyFlags);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mrA.size + mrR.size + 65536, (uint32_t)mt };
    VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, 0, &mem));
    VkDeviceSize offR = (mrA.size + 65535) & ~65535ull;
    CHECK(vkBindBufferMemory(dev, A, mem, 0)); CHECK(vkBindBufferMemory(dev, R, mem, offR));
    void *p; CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p));
    uint32_t *w32 = p; for (size_t i = 0; i < SIZE / 4; i++) w32[i] = (uint32_t)(i * 2654435761u);

    VkDescriptorSetLayoutBinding b[2] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0 },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0 } };
    VkDescriptorSetLayoutCreateInfo dsl = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 2, b };
    VkDescriptorSetLayout sl; CHECK(vkCreateDescriptorSetLayout(dev, &dsl, 0, &sl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &sl, 1, &pcr };
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, 0, &pl));
    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, (size_t)spv_size, spv };
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smi, 0, &sm));
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 1, &ps };
    VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpi, 0, &dp));
    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, dp, 1, &sl };
    VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo dbi[2] = { { A, 0, SIZE }, { R, 0, 64 << 20 } };
    VkWriteDescriptorSet wr[2] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &dbi[0] },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &dbi[1] } };
    vkUpdateDescriptorSets(dev, 2, wr, 0, 0);

    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qf };
    VkCommandPool cp; CHECK(vkCreateCommandPool(dev, &cpi, 0, &cp));
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CHECK(vkCreateFence(dev, &fci, 0, &fence));

    // WG size (spec constant 0) x chunk (vec4s per workgroup, push constant) x loads per thread per step (spec constant 1)
    uint32_t wgs[] = { 32, 64, 128, 256 }, vecs[] = { 1, 2, 4 };
    uint32_t chunks_kib[] = { 16, 64, 256, 1024 };
    for (int iw = 0; iw < 4; iw++) for (int iv = 0; iv < 3; iv++) {
        uint32_t spec[2] = { wgs[iw], vecs[iv] };
        VkSpecializationMapEntry se[2] = { { 0, 0, 4 }, { 1, 4, 4 } };
        VkSpecializationInfo si = { 2, se, 8, spec };
        VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", &si }, pl };
        VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, 0, 1, &cpci, 0, &pipe));
        for (int ic = 0; ic < 4; ic++) {
            uint32_t chunk_vec4 = chunks_kib[ic] * 1024 / 16;
            uint32_t groups = (uint32_t)(SIZE / 16 / chunk_vec4);
            uint32_t pc[2] = { chunk_vec4, 0 };
            double best = 1e9;
            for (int rep = 0; rep < 6; rep++) {
                VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
                vkResetCommandBuffer(cb, 0); CHECK(vkBeginCommandBuffer(cb, &bi));
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, 0);
                vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
                for (int k = 0; k < 4; k++) {   // four passes over the buffer per submission
                    vkCmdDispatch(cb, groups, 1, 1);
                    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
                }
                CHECK(vkEndCommandBuffer(cb));
                VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb };
                double t0 = now();
                CHECK(vkQueueSubmit(q, 1, &sub, fence));
                CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull));
                double dt = now() - t0;
                vkResetFences(dev, 1, &fence);
                if (dt < best) best = dt;
            }
            printf("wg %3u  loads %u x16B  chunk %4u KiB : %6.1f GB/s\n", wgs[iw], vecs[iv], chunks_kib[ic], 4.0 * SIZE / best / 1e9);
        }
        vkDestroyPipeline(dev, pipe, 0);
    }
    return 0;
}
