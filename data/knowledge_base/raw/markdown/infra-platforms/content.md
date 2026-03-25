# AI Server Platform Design — Technical Reference

AI server design requires deliberate choices across CPU, memory, interconnect, power, and form factor. Each component constrains or enables the GPU accelerators that drive the actual workload. Getting the host platform wrong creates bottlenecks that no amount of GPU hardware can overcome.

## CPU Selection

The two viable server CPU families for AI infrastructure are AMD EPYC and Intel Xeon Scalable. AMD EPYC Genoa (Zen 4, 9004 series) offers up to 96 cores per socket, 12 DDR5 memory channels, and 128 PCIe Gen5 lanes in a single-socket configuration. The newer Turin (Zen 5, 9005 series) extends core counts to 192 with dense cores optimized for throughput. Intel Xeon Sapphire Rapids (4th Gen) provides up to 60 cores, 8 DDR5 channels, and 80 PCIe Gen5 lanes per socket; Emerald Rapids (5th Gen) improves clocks and cache but keeps the same platform. For AI servers, the critical CPU metrics are PCIe lane count (determines how many GPUs and NICs can attach at full bandwidth), memory channel count (determines data loading throughput), and core count (determines preprocessing and data pipeline parallelism). Single-socket EPYC designs are increasingly popular because they provide enough lanes and channels without the NUMA complexity of dual-socket systems.

## Memory Architecture

Host DDR5 memory serves data loading, preprocessing pipelines, host-side buffers, gradient synchronization buffers, and operating system requirements. Each DDR5 channel provides roughly 38-51 GB/s depending on DIMM speed (DDR5-4800 to DDR5-6400). A 12-channel EPYC system with one DIMM per channel delivers approximately 460-614 GB/s aggregate host memory bandwidth. ECC is mandatory for production servers. The practical rule of thumb is to provision 2-4x the total GPU memory in host RAM. An 8x H100 SXM server with 640 GB of GPU memory should have 1-2 TB of host DDR5 to avoid stalling data loaders and preprocessing stages. Undersized host memory is one of the most common causes of GPU underutilization in training workloads.

## PCIe Topology and NVLink

PCIe is the baseline host-to-device interconnect. PCIe Gen4 x16 provides 32 GB/s bidirectional; PCIe Gen5 x16 doubles that to 64 GB/s (128 GB/s bidirectional when counting both directions). CPU-to-GPU data transfers, NIC communication, and NVMe storage all share the PCIe fabric. PCIe switch chips (historically PLX/Broadcom) are used in multi-GPU servers to fan out limited CPU root ports to more devices, but each switch hop adds latency and can become a bandwidth bottleneck.

NVLink is NVIDIA's proprietary GPU-to-GPU interconnect that bypasses PCIe entirely. NVLink 4.0 (Hopper) provides 900 GB/s bidirectional bandwidth per GPU — roughly 7x the bandwidth of PCIe Gen5 x16. NVLink matters for workloads that require frequent GPU-to-GPU communication: tensor parallelism, pipeline parallelism, large all-reduce operations in data parallel training, and any collective that moves gradients or activations between GPUs. For inference workloads with independent per-GPU batches, PCIe connectivity is often sufficient.

## NVSwitch and Full-Mesh Connectivity

In HGX and DGX systems, NVSwitch provides a full-mesh all-to-all GPU interconnect fabric. The Hopper-generation NVSwitch connects all 8 GPUs with equal bandwidth to any other GPU in the baseboard, eliminating the hierarchical NVLink topologies of earlier generations. This is critical for large model training where tensor parallel communication patterns require uniform bisection bandwidth. DGX H100 systems include 4 NVSwitch chips providing 900 GB/s per GPU to the switch fabric.

## Power, Cooling, and Rack Density

Modern GPU TDPs range from 300W (L40S, PCIe) to 700W (H100 SXM). An 8x H100 SXM server draws approximately 10-11 kW total including CPUs, memory, fans, and power supply losses. Rack power densities for GPU clusters commonly reach 30-50 kW per rack, far exceeding the 8-15 kW typical of traditional data center racks. H100 SXM systems strongly benefit from direct liquid cooling (DLC) to manage the 700W per GPU thermal load; air cooling is possible but requires significant airflow and limits rack density. Facility planning for power distribution, cooling capacity, and floor weight loading must happen before hardware procurement.

## Form Factors

Three primary form factors exist for AI servers. DGX systems are NVIDIA's integrated, validated reference platforms with NVLink, NVSwitch, networking, and software pre-configured. HGX baseboards provide the same 8-GPU NVLink/NVSwitch module to OEM partners (Dell, HPE, Lenovo, Supermicro) who build complete servers around them. PCIe-based servers use standard 2U or 4U rackmount chassis with GPUs in PCIe slots, suitable for inference (4x L40S) or smaller training configurations where NVLink is not required.

## Practical Sizing Examples

A training server for large model development typically uses 8x H100 SXM (HGX baseboard), single-socket EPYC Genoa, 2 TB DDR5, 4x ConnectX-7 InfiniBand NDR (400 Gb/s each) for inter-node communication, and local NVMe for scratch. An inference server might use 4x L40S in a standard PCIe 4U chassis, dual-socket Xeon for broader PCIe lane availability, 512 GB DDR5, and 2x ConnectX-7 Ethernet for client traffic. The motherboard's GPU slot topology, PCIe bifurcation support, and network card slots (particularly for ConnectX-7 InfiniBand NDR adapters) determine what configurations are physically possible.
