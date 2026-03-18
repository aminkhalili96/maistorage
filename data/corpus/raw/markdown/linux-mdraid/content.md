# Linux mdraid and RAID Configuration for AI Servers

Linux software RAID (mdraid) provides kernel-level RAID functionality managed by the `mdadm` utility. In AI server deployments, mdraid configures local storage for OS redundancy, high-speed scratch space, and dataset staging. Unlike hardware RAID controllers, mdraid requires no proprietary firmware, supports online capacity expansion, and provides full visibility into array health through standard Linux interfaces.

## mdadm Fundamentals

Create arrays with `mdadm --create /dev/md0 --level=10 --raid-devices=4 /dev/nvme{0..3}n1`. Assemble existing arrays with `mdadm --assemble /dev/md0 /dev/nvme{0..3}n1`. Monitor health with `mdadm --detail /dev/md0`. Generate configuration with `mdadm --detail --scan >> /etc/mdadm/mdadm.conf`, then regenerate initramfs via `update-initramfs -u` (Debian/Ubuntu) or `dracut --force` (RHEL/Rocky).

## RAID Levels for AI Servers

**RAID 0 (Striping)**: Stripes data across all drives with no redundancy. A 4-drive NVMe RAID 0 delivers approximately 4x single-drive throughput. Use exclusively for ephemeral scratch space where data loss is acceptable. Create: `mdadm --create /dev/md0 --level=0 --raid-devices=4 /dev/nvme{0..3}n1 --chunk=256K`

**RAID 1 (Mirroring)**: Mirrors data across two drives at 50% capacity efficiency. Use for OS boot drives on every AI server to ensure boot survivability after a single drive failure. Create: `mdadm --create /dev/md1 --level=1 --raid-devices=2 /dev/sda1 /dev/sdb1`

**RAID 5 (Single Parity)**: Single-drive fault tolerance with `(N-1)/N` efficiency. Avoid for drives larger than 4 TB -- rebuild times of 12-48 hours create dangerous vulnerability windows where a second failure causes total loss.

**RAID 6 (Double Parity)**: Tolerates any two simultaneous drive failures. Capacity efficiency is `(N-2)/N`. Suitable for large-capacity archival pools where write throughput is secondary to data protection. Create: `mdadm --create /dev/md0 --level=6 --raid-devices=6 /dev/sd{a..f}`

**RAID 10 (Mirrored Stripes)**: Best balance of performance and redundancy for AI workloads -- near-RAID 0 sequential throughput, excellent random I/O, tolerates at least one drive failure per mirrored pair. 50% capacity efficiency. Create: `mdadm --create /dev/md0 --level=10 --raid-devices=4 /dev/nvme{0..3}n1 --layout=f2`. The `f2` (far 2) layout distributes mirrors across non-adjacent drives for improved sequential reads; `n2` (near 2, default) is better for random I/O patterns typical of dataset loading.

## Stripe Size Tuning

The `--chunk` parameter sets the stripe unit (default 512K). For sequential I/O (checkpoints, large datasets), use 256K-1M. For random I/O (image classification datasets), use 64K-128K. Align the filesystem: `mkfs.xfs -d su=256k,sw=3 /dev/md0` where `su` is the stripe unit and `sw` is the stripe width (data drives, excluding parity).

## NVMe RAID Considerations

NVMe drives bypass storage controller bottlenecks via direct PCIe attachment. Software RAID is the standard approach because most hardware RAID controllers do not support NVMe. CPU overhead for mdraid parity is minimal on processors with AVX-512. Do not use hardware RAID controllers for NVMe -- they add latency without benefit. ZFS (`zpool create`) is an alternative offering checksumming and compression at higher memory cost (1 GB RAM per TB guideline).

## AI Server Storage Architecture

| Mount Point | RAID Level | Drives | Purpose |
|---|---|---|---|
| `/` and `/boot` | RAID 1 | 2x 480GB SATA SSD | OS, boot |
| `/scratch` | RAID 0 | 4x 3.84TB NVMe | Training scratch, shuffle buffer |
| `/data` | RAID 10 | 4x 3.84TB NVMe | Dataset staging from network storage |
| `/checkpoints` | Single NVMe or NFS | 1x NVMe or NFS mount | Checkpoint durability via remote storage |

## Monitoring and Alerting

- **Real-time**: `cat /proc/mdstat` shows sync progress and array state
- **Detailed**: `mdadm --detail /dev/md0` reports state, member drives, spare count
- **Email alerts**: Configure `MAILADDR` in `/etc/mdadm/mdadm.conf`, run `mdadm --monitor --scan --daemonise`
- **SMART**: `smartctl -a /dev/nvme0n1` for drive health; schedule tests via `/etc/smartd.conf`

## Rebuild Times and Tuning

Default rebuild speed limits: `/proc/sys/dev/raid/speed_limit_min` (1000 KB/s) and `speed_limit_max` (200000 KB/s). Increase minimum to reduce vulnerability: `echo 50000 > /proc/sys/dev/raid/speed_limit_min`. A 2 TB RAID 10 rebuild at 200 MB/s takes approximately 2.8 hours; RAID 5/6 parity recalculation multiplies this by 2-3x. Configure hot spares with `mdadm --add /dev/md0 /dev/nvme4n1` for automatic failover.

## Filesystem Choices

**XFS**: Recommended for AI data volumes. Handles large files efficiently, supports online growth (`xfs_growfs`), excellent parallel I/O. Mount: `mount -o noatime,discard,allocsize=64m /dev/md0 /data`. **ext4**: Suitable for OS boot drives. Mount with `noatime,discard`. Maximum file size is 16 TB (vs. 8 EB for XFS).

## Common Failures and Recovery

**Degraded array**: Identify failed drive via `mdadm --detail /dev/md0`. Remove: `mdadm --remove /dev/md0 /dev/nvme2n1`. Replace hardware, then add: `mdadm --add /dev/md0 /dev/nvme2n1`. Rebuild starts automatically. **Growing arrays**: `mdadm --grow /dev/md0 --raid-devices=6 --add /dev/nvme4n1 /dev/nvme5n1` reshapes online without data loss.

## Integration with GPUDirect Storage

GPUDirect Storage (GDS) enables direct DMA between NVMe and GPU memory. RAID 0 provides best GDS compatibility; RAID 5/6 parity computation requires CPU involvement, partially negating the bypass benefit. For maximum GDS throughput, present individual NVMe drives directly to the `nvidia-fs` kernel module and configure allowed paths in `/etc/cufile.json`. Use mdraid only for non-GDS workloads.

## Best Practices

- RAID 1 for OS drives on every server without exception
- Avoid RAID 5 with drives larger than 4 TB
- Monitor SMART attributes proactively; replace drives showing `Reallocated_Sector_Ct` increases
- Configure hot spares for all production arrays
- Use `noatime` mount option on all data volumes
- Align stripe width with filesystem block allocation
- Back up `/etc/mdadm/mdadm.conf` off-node
