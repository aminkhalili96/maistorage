# NVIDIA RAPIDS -- GPU-Accelerated Data Science

RAPIDS is NVIDIA's suite of open-source, GPU-accelerated libraries for data science and analytics. It provides pandas-like DataFrames (cuDF), scikit-learn-compatible machine learning (cuML), and graph analytics (cuGraph) on NVIDIA GPUs, achieving 10-100x speedups over CPU-based tools and enabling interactive exploration on datasets that previously required distributed CPU clusters.

## Core Libraries

RAPIDS consists of several interoperable libraries covering the full data science workflow:

- **cuDF:** GPU-accelerated DataFrames with a pandas-compatible API. Reads CSV, Parquet, ORC, JSON, and Avro at GPU memory bandwidth. Supports groupby, join, merge, window functions, string operations, and datetime processing on GPU.

- **cuML:** GPU-accelerated machine learning with a scikit-learn-compatible API. Implements linear regression, logistic regression, random forest, k-means, DBSCAN, UMAP, PCA, t-SNE, KNN, and SVM. Models can be exported and used interchangeably with scikit-learn.

- **cuGraph:** GPU-accelerated graph analytics including PageRank, BFS, SSSP, connected components, community detection (Louvain, Leiden), and Jaccard similarity. Handles graphs with billions of edges on a single GPU.

- **cuSpatial:** Geospatial analytics including point-in-polygon, spatial joins, and trajectory analysis.

- **cuxfilter:** Cross-filtering visualization framework for interactive dashboards.

All libraries share a GPU memory model based on Apache Arrow columnar format, enabling zero-copy data exchange between them.

## cuDF for Data Preprocessing

cuDF serves as the primary preprocessing engine in GPU-accelerated pipelines. Its `cudf.pandas` accelerator mode acts as a drop-in pandas replacement, automatically routing operations to GPU when beneficial and falling back to CPU for unsupported operations.

Key capabilities:
- **File I/O:** Parquet reading at 5-15 GB/s (vs. 0.5-1.5 GB/s on CPU). CSV parsing at 2-5 GB/s.
- **ETL operations:** Filter, join, groupby, and window functions on GPU. String processing (regex, split, replace) runs via `cudf.core.column.string`.
- **Multi-GPU scaling:** Dask-cuDF extends DataFrames across multiple GPUs and nodes, enabling ETL on datasets larger than single-GPU memory.

## cuML for Feature Engineering

cuML accelerates feature engineering and preprocessing steps before model training:

- **Preprocessing:** StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder run directly on GPU memory with no CPU data movement.
- **Dimensionality reduction:** PCA, UMAP, and t-SNE on GPU. GPU-accelerated UMAP reduces million-point datasets in seconds rather than hours.
- **Clustering:** K-means, DBSCAN, HDBSCAN on GPU for data cleaning, anomaly detection, and unsupervised feature discovery.

Since cuML runs on the same GPU used for training, preprocessed feature tensors remain in GPU memory throughout the pipeline, eliminating CPU-GPU transfer bottlenecks.

## Integration with Training Pipelines

RAPIDS integrates with deep learning frameworks through zero-copy data exchange:

- **RAPIDS to PyTorch:** cuDF DataFrames convert to PyTorch tensors via DLPack (`torch.from_dlpack(cudf_series.to_dlpack())`), keeping data in GPU memory throughout.
- **RAPIDS to TensorFlow:** DLPack-based zero-copy path or conversion through CuPy arrays.
- **Custom data loaders:** cuDF serves as backend for PyTorch Dataset/DataLoader implementations, performing all data reading and transformation on GPU before feeding training batches.

A typical pipeline: raw Parquet files read by cuDF, feature engineering in cuML (scaling, encoding, reduction), resulting GPU tensors flow directly into a PyTorch training loop with no CPU-GPU transfers.

## Deployment

RAPIDS is available through multiple channels:

- **NGC containers:** Pre-built Docker containers on NGC (`nvcr.io/nvidia/rapidsai/rapidsai`) with RAPIDS, CUDA, and all dependencies.
- **Conda:** `conda install -c rapidsai -c conda-forge -c nvidia cudf cuml cugraph`
- **pip:** `pip install cudf-cu12 cuml-cu12` for CUDA 12.x environments.
- **Requirements:** CUDA 11.2+ (recommended 12.x), NVIDIA GPU with Pascal or later (compute capability 6.0+), Linux.

Multi-GPU deployments use Dask-CUDA for scheduling and UCX for high-performance inter-GPU communication over NVLink and InfiniBand.

## AI Cluster Relevance

**Data preprocessing acceleration:** RAPIDS eliminates preprocessing bottlenecks when data transformation (not storage I/O or model compute) is the constraint. Common in tabular workloads, recommendation systems, and feature engineering pipelines.

**Feature store computation:** GPU-accelerated feature computation for real-time and batch feature stores at interactive speeds.

**Log analytics and cluster monitoring:** cuDF processes gigabytes of system logs, DCGM metrics, and scheduler data at interactive speeds. cuGraph analyzes communication patterns in multi-node training to identify network bottlenecks.

**Data quality and validation:** GPU-accelerated data profiling and anomaly detection on training datasets. cuML's DBSCAN and isolation forest identify outliers in high-dimensional feature spaces at scale.
