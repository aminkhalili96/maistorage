# Slurm Workload Manager for GPU Clusters

Slurm (Simple Linux Utility for Resource Management) is the dominant job scheduler for HPC and GPU training clusters. It provides deterministic resource allocation, job queuing, and accounting across thousands of nodes, making it the standard choice for multi-node distributed AI training workloads.

## Architecture

Slurm follows a centralized controller model with four core daemons:

- **slurmctld** — the central controller daemon running on the head node. It manages the job queue, schedules resources, and tracks cluster state. Runs in active/passive HA with a backup controller configured via `BackupController` in `slurm.conf`.
- **slurmd** — the compute node daemon running on every worker. It launches and monitors jobs, enforces resource limits via cgroups, and reports node health back to slurmctld.
- **slurmdbd** — the database daemon that records job accounting, usage statistics, and fair-share data. Typically backed by MySQL/MariaDB. Required for fair-share scheduling and `sacct` historical queries.
- **slurmrestd** — the REST API daemon (Slurm 20.02+) exposing job submission, queue status, and node info over HTTP/JSON. Useful for integrating Slurm with web portals, CI/CD pipelines, and MLOps platforms.

Configuration lives primarily in `/etc/slurm/slurm.conf` on all nodes, with `gres.conf` for GPU definitions and `cgroup.conf` for resource isolation.

## Partitions and QOS

Partitions segment the cluster into logical groups of nodes with distinct access policies. A partition defines which users or accounts can submit jobs, default and maximum time limits, node lists, and priority tiers.

```
PartitionName=gpu-a100 Nodes=gpu[001-064] Default=NO MaxTime=72:00:00 State=UP
PartitionName=gpu-h100 Nodes=gpu[065-128] Default=NO MaxTime=48:00:00 State=UP AllowQos=high,urgent
```

Quality of Service (QOS) layers additional controls on top of partitions: priority boost, preemption rights, job limits per user, and GrpTRES caps. Preemption policies (`PreemptType=preempt/qos`, `PreemptMode=REQUEUE`) allow urgent jobs to reclaim GPUs from lower-priority work, which is critical when sharing expensive GPU nodes between research and production training.

## GRES (Generic Resources) for GPUs

Slurm tracks GPUs as Generic Resources (GRES). Each node declares its GPU inventory in `/etc/slurm/gres.conf`:

```
NodeName=gpu001 Name=gpu Type=a100 File=/dev/nvidia[0-7]
NodeName=gpu065 Name=gpu Type=h100 File=/dev/nvidia[0-7]
```

Users request GPUs with `--gres=gpu:a100:4` to get exactly 4 A100 GPUs. Setting `AutoDetect=nvml` in `gres.conf` lets Slurm auto-discover GPU types and counts via the NVIDIA Management Library, eliminating manual enumeration. GPU binding options (`--gpu-bind=closest`, `--gpu-bind=map_gpu`) control which GPUs a task sees, and `--gpus-per-task` ensures each MPI rank gets its own GPU.

## Job Submission

Slurm provides three job submission commands:

- **`sbatch`** — submits a batch script to the queue. The script contains `#SBATCH` directives for resource requests. This is the standard method for training jobs.
- **`srun`** — launches a job step, either interactively or inside an `sbatch` script. It is also the preferred MPI launcher under Slurm, replacing `mpirun` in most GPU cluster deployments.
- **`salloc`** — allocates resources and starts an interactive shell on the allocated nodes. Useful for debugging multi-GPU jobs or running Jupyter notebooks on GPU nodes.

A typical multi-node training submission:

```bash
sbatch --job-name=llm-pretrain --nodes=8 --ntasks-per-node=8 \
       --gres=gpu:h100:8 --cpus-per-task=12 --mem=0 \
       --partition=gpu-h100 --time=48:00:00 train.sh
```

## Resource Scheduling

Slurm's scheduler uses a priority-based queue with backfill. The multi-factor priority formula combines: job age, fair-share (historical usage vs allocation), job size, partition priority, and QOS priority. Weights are configured via `PriorityWeightAge`, `PriorityWeightFairshare`, etc. in `slurm.conf`.

Backfill scheduling (`SchedulerType=sched/backfill`) allows smaller jobs to start ahead of higher-priority jobs if they will complete before the reserved resources are needed. This dramatically improves cluster utilization, which is critical when GPU-hours cost $2-3/hr per device.

## MPI Integration

For distributed training across multiple nodes, Slurm integrates with MPI via PMIx (Process Management Interface for Exascale). Configure with `MpiDefault=pmix_v4` in `slurm.conf`. Using `srun` as the MPI launcher instead of `mpirun` ensures proper GPU affinity, cgroup enforcement, and signal propagation. A typical PyTorch distributed training launch:

```bash
srun --mpi=pmix_v4 python -m torch.distributed.run \
     --nproc_per_node=8 --nnodes=$SLURM_NNODES \
     --node_rank=$SLURM_NODEID train.py
```

## GPU Isolation and Cgroups

Slurm enforces GPU isolation through cgroups. In `cgroup.conf`, set `ConstrainDevices=yes` to restrict each job to only its allocated GPUs via the devices cgroup controller. This prevents jobs from seeing or accessing GPUs allocated to other users. Combined with `TaskPlugin=task/cgroup,task/affinity`, Slurm also binds tasks to specific CPU cores and NUMA domains adjacent to their allocated GPUs, minimizing PCIe traversal latency.

## Monitoring and Accounting

- **`squeue`** — view the job queue, filter by user, partition, or state
- **`sacct`** — query historical job records from slurmdbd (elapsed time, MaxRSS, GPU utilization via TRES)
- **`sinfo`** — display node and partition status, idle/allocated/drained states
- **Job arrays** (`--array=0-99`) — submit 100 variations of a hyperparameter sweep as a single job
- **Job dependencies** (`--dependency=afterok:12345`) — chain jobs so fine-tuning starts only after pretraining succeeds

## Slurm vs Kubernetes for AI Workloads

Slurm excels at batch scheduling for multi-node training: deterministic GPU allocation, gang scheduling (all-or-nothing for distributed jobs), tight MPI integration, and backfill are mature and battle-tested at scale. Kubernetes is stronger for serving, microservice orchestration, and operator-driven lifecycle management. Many production AI platforms use both: Slurm for training, Kubernetes for inference serving and MLOps tooling. Projects like NVIDIA Enroot and Pyxis bridge the gap by running OCI containers under Slurm.
