# MinIO and S3-Compatible Object Storage for AI Infrastructure

MinIO is a high-performance, S3-compatible object storage system designed for large-scale data infrastructure. In AI environments it serves as the durable storage layer for datasets, model checkpoints, training artifacts, and data lake workloads. MinIO can be deployed on bare metal, virtual machines, or Kubernetes and is released under the GNU AGPLv3 license.

## Architecture and Erasure Coding

MinIO operates in standalone (single node) or distributed (multiple nodes, multiple drives per node) modes. Production AI deployments use distributed mode exclusively. A distributed cluster requires a minimum of 4 nodes with at least 1 drive each; the recommended topology is 4-16 nodes with 4-16 NVMe drives per node. MinIO uses a deterministic hashing algorithm to distribute objects across erasure sets without a centralized metadata server.

Erasure coding uses Reed-Solomon algorithms to split each object into data and parity shards. The default ratio is EC:4 (4 parity shards). For a 16-drive erasure set with EC:4, the cluster tolerates losing up to 4 drives while maintaining full availability. Storage efficiency is `(N - parity) / N` -- 75% usable with EC:4 on 16 drives. Configure via `MINIO_STORAGE_CLASS_STANDARD` and `MINIO_STORAGE_CLASS_RRS`.

## S3-Compatible API

MinIO implements `CreateBucket`, `PutObject`, `GetObject`, `DeleteObject`, `ListObjectsV2`, multipart upload, presigned URLs, and bucket versioning. All S3 SDKs (boto3, AWS CLI, s3cmd) work without modification. Multipart upload supports part sizes from 5 MB to 5 GB with up to 10,000 parts; the recommended part size for checkpoint files is 64-128 MB. Bucket versioning creates immutable dataset snapshots -- every `PutObject` creates a new version, and object locking (WORM) prevents accidental deletion of training data.

## Deployment Patterns

**Standalone**: Single server for development only. **Distributed**: Multi-node with erasure coding, deployed via `minio server https://minio{1...4}.example.com/mnt/disk{1...4}`. **Multi-site replication**: Active-active replication via `mc replicate add`, asynchronous by default, essential for disaster recovery in production training environments.

## AI-Specific Use Cases

**Dataset versioning**: Store training data as versioned objects; record version IDs per training run for exact reproducibility. **Checkpoint storage**: Frameworks like PyTorch and Megatron-LM write periodic checkpoints via S3FS-FUSE at `/mnt/checkpoints` or through `fsspec`. MinIO sustains 10+ GB/s write throughput per node with NVMe drives, ensuring checkpoints do not stall training. **Data lake patterns**: Store Parquet/Arrow datasets accessed via `s3://bucket/prefix/` with PyArrow and Hive-style partitioning for pruning. **Model registry**: Organize model artifacts by name and version, tagged with custom metadata (`X-Amz-Meta-Model-Version`).

## Tiering and Lifecycle Management

ILM rules defined via `mc ilm rule add` transition objects between storage classes automatically:

- Hot tier (NVMe): active datasets and recent checkpoints (0-7 days)
- Warm tier (SSD): older checkpoints and experiment artifacts (7-90 days)
- Cold tier (HDD or remote S3): archived data (90+ days)

Objects in all tiers remain retrievable via the same S3 API path.

## Performance Tuning

Drive count is the primary throughput lever; each NVMe contributes 1-3 GB/s sequential throughput. Network must match: 25GbE minimum, 100GbE recommended for multi-node clusters. Objects smaller than 1 MB incur high per-request overhead; batch small files into tar archives. Set `MINIO_API_REQUESTS_MAX` to 1000+ for checkpoint bursts. Use JBOD configuration -- MinIO's erasure coding provides redundancy; layering RAID underneath wastes capacity.

## Security

Enable TLS via certificates in `${HOME}/.minio/certs/`. IAM-compatible policies control access scoped to bucket prefixes (`s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`). Encryption at rest via SSE-S3 (server-managed keys) or SSE-KMS (Vault, AWS KMS). For multi-tenant AI clusters, create one MinIO user per team with prefix-scoped policies.

## Integration with AI Frameworks

PyTorch DataLoader reads from MinIO via `fsspec` and `s3fs`: `torchdata.datapipes.iter.FSSpecFileOpener("s3://bucket/train/", mode="rb")`. FUSE-based access via `s3fs bucket /mnt/data -o url=https://minio.local:9000` simplifies code but adds kernel overhead; the S3 API is preferred for high-throughput pipelines.

## Monitoring

Prometheus metrics at `/minio/v2/metrics/cluster` expose `minio_s3_requests_total`, `minio_s3_traffic_received_bytes_total`, `minio_node_drive_free_bytes`, and `minio_node_drive_errors_total`. The MinIO Console (port 9001) provides a web dashboard for monitoring and administration.

## Object Storage vs. Parallel File Systems

Use MinIO for durable, versioned, long-term data (datasets, checkpoints, model artifacts). Use parallel file systems (Lustre, BeeGFS) for high-IOPS scratch during active training. Most large-scale clusters use both: parallel filesystem as the scratch tier with MinIO as the persistent layer, staging active datasets before training.

## Hardware Recommendations

- **Drives**: NVMe SSDs (Samsung PM9A3, Intel D7-P5520) for hot tier; SATA SSDs for warm; HDDs for cold
- **Network**: 25GbE minimum, 100GbE recommended; LACP bonding for redundancy
- **CPU**: 8-16 cores per node; erasure coding uses AVX-512 SIMD
- **Memory**: 32-64 GB per node for metadata caching
- **Configuration**: JBOD, no hardware RAID; format with XFS (`mkfs.xfs -f -L DISK1 /dev/nvme0n1`)
