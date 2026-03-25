# GPU Cluster Monitoring and Observability

Operating NVIDIA GPU clusters at scale requires an end-to-end monitoring stack that covers every layer of the infrastructure — from hardware health and thermal management to NVLink fabric performance and training job efficiency. A well-instrumented cluster enables operators to detect hardware faults before they cascade, identify underutilized resources for right-sizing, and give ML engineers visibility into training bottlenecks. This guide covers the standard monitoring stack used in production NVIDIA GPU deployments.

## DCGM Exporter

NVIDIA Data Center GPU Manager (DCGM) is the foundation of GPU telemetry. The DCGM Exporter runs as a lightweight process that queries GPU metrics via the DCGM library and exposes them in Prometheus format on port 9400. In Kubernetes environments, the DCGM Exporter is deployed as a DaemonSet, typically managed automatically by the NVIDIA GPU Operator.

Key metrics exported by DCGM:

- `DCGM_FI_DEV_GPU_UTIL` — GPU core utilization percentage (0-100%)
- `DCGM_FI_DEV_FB_USED` — Framebuffer (GPU memory) used in MiB
- `DCGM_FI_DEV_GPU_TEMP` — GPU die temperature in degrees Celsius
- `DCGM_FI_DEV_POWER_USAGE` — Current power draw in watts
- `DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL` — Aggregate NVLink throughput
- `DCGM_FI_DEV_XID_ERRORS` — GPU Xid error count (indicates hardware/driver faults)
- `DCGM_FI_DEV_SM_CLOCK` — Current streaming multiprocessor clock frequency in MHz
- `DCGM_FI_DEV_MEM_CLOCK` — Current memory clock frequency in MHz

Custom metric selection is configured via `/etc/dcgm-exporter/default-counters.csv`, where each line specifies a DCGM field ID to export. For interactive dashboards, a scrape interval of 5-15 seconds provides responsive visualizations. For long-term storage, 30-second intervals reduce storage costs while retaining sufficient granularity for trend analysis.

## Prometheus Configuration

Prometheus scrapes DCGM Exporter endpoints and stores GPU time-series data. In Kubernetes, a `ServiceMonitor` custom resource enables automatic discovery of DCGM Exporter pods, eliminating manual scrape target configuration as nodes are added or removed.

Recording rules should be defined for aggregated cluster-level metrics, such as average GPU utilization across all nodes, total framebuffer memory consumed, and per-node utilization summaries. These pre-computed aggregations reduce query latency for dashboard panels that span hundreds of GPUs.

GPU metric labels should be enriched with contextual metadata: node hostname, GPU index (0-7 for 8-GPU nodes), GPU model name (e.g., A100-SXM4-80GB), and driver version. This label enrichment enables filtering and grouping in dashboards and alert rules.

For retention, keep 15-30 days of raw metrics in Prometheus local storage. For long-term retention beyond 30 days, integrate with Thanos or Cortex to offload historical data to object storage while maintaining a unified query interface.

## Grafana Dashboards

Effective GPU cluster dashboards are organized in three tiers of detail:

**Cluster overview** shows the fleet-wide picture: total GPU count and availability, a utilization heatmap across all nodes, aggregate memory pressure, and overall power consumption versus facility capacity.

**Per-node view** drills into individual servers with an 8-GPU utilization chart (one time-series per GPU), NVLink bandwidth between GPU pairs, and PCIe host-to-device throughput. This view quickly identifies asymmetric GPU loads or degraded interconnects.

**Per-job view** requires integration with Slurm job IDs or Kubernetes pod labels to correlate GPU metrics with specific training runs. Panels show GPU utilization per rank, memory consumption per rank, and communication overhead as a fraction of total step time.

Recommended panel types include: GPU utilization histogram (distribution across the fleet), memory waterfall chart (allocation over time), temperature trend lines, power draw versus TDP limits, and NVLink bandwidth per link for diagnosing interconnect bottlenecks.

## Alerting Rules

Alerting rules should be tiered by severity to avoid alert fatigue:

**Critical** alerts require immediate operator response: GPU Xid errors detected (indicates hardware fault or driver crash), GPU temperature exceeding 85 degrees Celsius (thermal throttling imminent), uncorrectable ECC memory errors, and NVLink link failure (interconnect down between GPU pairs).

**Warning** alerts indicate conditions that need attention within hours: GPU utilization below 20% for more than 30 minutes (resource waste), GPU memory utilization above 95% (OOM crash risk), GPU clock throttling detected (power or thermal limiting), and sustained power draw exceeding the configured power limit.

**Info** alerts track configuration drift: NVIDIA driver version mismatch across cluster nodes, CUDA toolkit version inconsistency between nodes, and firmware version discrepancies that may affect NVLink or NVSwitch behavior.

## Capacity Planning

Monitoring data feeds directly into capacity planning decisions. Track GPU-hours consumed per team, project, or cost center to enable internal chargeback and budgeting. Analyze utilization trends over weeks and months to inform procurement timing — sustained utilization above 80% across the cluster signals the need for expansion.

Queue wait time in Slurm or Kubernetes scheduling delay serves as a demand signal independent of utilization. Even moderate average utilization can mask severe contention if jobs queue for hours during peak periods.

Right-sizing analysis identifies jobs that consistently use less than 50% of GPU memory, suggesting they could run on smaller GPU models or leverage MIG (Multi-Instance GPU) partitioning on A100 or H100 hardware. For budget forecasting, apply linear regression on GPU-hour growth trends to project infrastructure costs 6-12 months ahead.

## Job-Level Observability

Correlating GPU hardware metrics with job-level telemetry provides the deepest insight into training efficiency. In Slurm environments, `sacct` job accounting records (start time, end time, allocated GPUs) can be joined with DCGM time-series data to compute per-job GPU utilization and memory efficiency.

In Kubernetes, the Kubeflow Training Operator exposes per-job metrics including replica status, restart counts, and completion times. These integrate with Prometheus via standard ServiceMonitor configuration.

Training-specific metrics require instrumentation within the training script: samples processed per second, loss curve values, gradient norm (for detecting instability), and the communication-to-computation ratio (time spent in NCCL collectives versus forward/backward passes). PyTorch Profiler can export detailed execution traces to TensorBoard or Perfetto for kernel-level analysis of GPU occupancy and memory allocation patterns.

## Network Monitoring

High-performance GPU clusters rely on InfiniBand and NVLink fabrics whose health is critical to training throughput. Standard InfiniBand diagnostics include `ibstat` for port status, `perfquery` for per-port traffic counters, and `ibdiagnet` for fabric-wide topology validation and error scanning.

RDMA traffic counters per port (bytes transmitted, bytes received, packet errors) should be collected and exported to Prometheus. NVIDIA Unified Fabric Manager (UFM) provides a centralized view of InfiniBand fabric topology, health status, and performance across the entire cluster, with built-in alerting for link flaps and error thresholds.

NVLink health is monitored via `nvidia-smi nvlink --status`, which reports per-link error counters, replay counts, and recovery events. Persistent NVLink CRC errors or replay events indicate cable or connector degradation requiring physical inspection.

## Common Monitoring Stack

The standard production monitoring stack for NVIDIA GPU clusters consists of Prometheus for metrics storage, DCGM Exporter for GPU telemetry, Grafana for visualization, and Alertmanager for notification routing.

In Kubernetes environments, the `kube-prometheus-stack` Helm chart deploys Prometheus, Grafana, and Alertmanager with sensible defaults. Add the DCGM Exporter DaemonSet (installed automatically by the GPU Operator or deployed manually) and configure a ServiceMonitor for scrape discovery.

In Slurm-managed bare-metal clusters, deploy `slurm-exporter` for job queue and accounting metrics alongside DCGM Exporter on every compute node. A Prometheus instance on a management node scrapes both exporters.

Optional additions to the stack include Loki for centralized log aggregation (GPU driver logs, Slurm controller logs, training stdout/stderr), and Tempo for distributed tracing of multi-service inference pipelines. Together, these tools provide metrics, logs, and traces — the three pillars of observability — across the GPU cluster.
