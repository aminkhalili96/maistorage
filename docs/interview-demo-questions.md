# Interview Demo Questions — Agentic RAG Quality Test

## How to use this file
Ask these questions during the demo in order. Each section tests a different capability.
The "Look for" notes tell you what makes a good answer. The "Red flags" tell you what
signals a bad answer. The "Talking point" is what to say to the interviewer.

Time budget: ~3 minutes per category, pick 1-2 questions per category.

---

## Category 1: Multi-Source Synthesis (show agentic reasoning)

### Q: "I'm scaling training from 4 to 8 GPUs and seeing poor scaling. What NCCL parameters should I tune, and how does NVLink vs PCIe affect this?"
- **Expected mode**: corpus-backed
- **Expected sources**: nccl, fabric-manager, dl-performance
- **Look for**:
  - Mentions specific NCCL environment variables (NCCL_ALGO, NCCL_PROTO, NCCL_P2P_LEVEL or similar)
  - Explains that NVLink provides higher bandwidth and lower latency than PCIe for GPU-to-GPU communication
  - Discusses the compute-to-communication ratio and how it degrades with more GPUs
- **Red flags**:
  - Generic advice with no specific NCCL parameters
  - Fails to distinguish NVLink vs PCIe impact on collective operations
- **Talking point**: "Notice how the system pulls from both the NCCL docs and the Fabric Manager guide to give a complete answer — it classified this as a multi-source distributed training question and searched across source families."

### Q: "How does Transformer Engine handle FP8 training, and when should I prefer BF16 instead?"
- **Expected mode**: corpus-backed
- **Expected sources**: transformer-engine, dl-performance
- **Look for**:
  - Explains FP8 forward/backward pass precision management in Transformer Engine
  - Describes when BF16 is safer (numerical stability, smaller models, fine-tuning)
  - Mentions Tensor Core utilization differences
- **Red flags**:
  - Confuses FP8 with INT8 quantization (inference vs training)
  - No mention of Transformer Engine's automatic mixed precision handling
- **Talking point**: "This question forces the system to synthesize across Transformer Engine and Deep Learning Performance docs — two different source families. The citations panel shows exactly which chunks contributed."

### Q: "What are the key differences between Megatron-Core parallelism strategies and when should I use tensor vs pipeline parallelism?"
- **Expected mode**: corpus-backed
- **Expected sources**: megatron-core, dl-performance, nemo-performance
- **Look for**:
  - Distinguishes tensor parallelism (splits layers) from pipeline parallelism (splits stages)
  - Mentions that tensor parallelism works best within a node (high bandwidth NVLink) while pipeline parallelism scales across nodes
  - References model size and memory constraints as decision factors
- **Red flags**:
  - Conflates data parallelism with tensor/pipeline parallelism
  - No mention of communication overhead tradeoffs between strategies
- **Talking point**: "This is a question MaiStorage customers ask when sizing multi-node training clusters. The system correctly retrieves from Megatron-Core and cross-references with the performance guides."

---

## Category 2: Deployment Pipeline (close the JD gap)

### Q: "Walk me through optimizing a trained PyTorch model for production inference on H100 GPUs using TensorRT."
- **Expected mode**: corpus-backed
- **Expected sources**: tensorrt, h100
- **Look for**:
  - Describes the export pipeline: PyTorch to ONNX to TensorRT engine
  - Mentions layer fusion, kernel auto-tuning, and precision calibration (FP16/INT8)
  - References H100-specific features like FP8 support or Hopper architecture advantages
- **Red flags**:
  - Describes only training workflows, not inference optimization
  - No mention of TensorRT's core optimization techniques (fusion, quantization)
- **Talking point**: "This question directly maps to MaiStorage's value proposition — customers buy H100 systems and need to know the full path from trained model to production serving. Notice the response mode is corpus-backed with citations."

### Q: "How do you configure Triton Inference Server to serve multiple models with dynamic batching?"
- **Expected mode**: corpus-backed
- **Expected sources**: triton-inference-server
- **Look for**:
  - Explains model repository structure and config.pbtxt
  - Describes dynamic batching configuration parameters (preferred batch sizes, max queue delay)
  - Mentions model concurrency and instance groups
- **Red flags**:
  - Vague answer without Triton-specific configuration details
  - Confuses Triton with other serving frameworks
- **Talking point**: "Triton is the standard for NVIDIA inference serving. The system retrieves specific configuration guidance from the Triton docs, not generic inference advice."

---

## Category 3: Infrastructure Decision-Making (show domain expertise)

### Q: "We're setting up a new AI training cluster with 32 GPUs. Should we use Slurm or Kubernetes for job scheduling, and what are the tradeoffs?"
- **Expected mode**: corpus-backed
- **Expected sources**: slurm-workload-manager, kubernetes-workloads, gpu-operator, infra-cluster-ops
- **Look for**:
  - Identifies Slurm as the traditional HPC scheduler with better GPU-aware scheduling for training
  - Identifies Kubernetes as better for mixed inference/training workloads and microservices
  - Discusses the GPU Operator's role in Kubernetes GPU management
- **Red flags**:
  - One-sided recommendation without discussing tradeoffs
  - No mention of GPU-specific scheduling considerations
- **Talking point**: "This is exactly the kind of architecture question MaiStorage's solution engineers face. The system synthesizes across our offline Slurm, Kubernetes, and GPU Operator sources to give a balanced recommendation."

### Q: "What storage architecture should we use for a training cluster that needs to serve large datasets and save checkpoints frequently?"
- **Expected mode**: corpus-backed
- **Expected sources**: gpudirect-storage, lustre, beegfs, infra-storage
- **Look for**:
  - Recommends a parallel file system (Lustre or BeeGFS) for high-throughput dataset serving
  - Mentions GPUDirect Storage for bypassing CPU on data loads
  - Discusses tiered storage (fast NVMe for checkpoints, object storage for cold data)
- **Red flags**:
  - Recommends only NFS or local disk without mentioning parallel file systems
  - No mention of GPUDirect Storage or I/O bottleneck considerations
- **Talking point**: "Storage architecture is MaiStorage's core business. The system correctly identifies the parallel file system tier plus GPUDirect Storage, which is exactly what their customers need to hear."

### Q: "What is DCGM and how should we use it to monitor GPU health in a production cluster?"
- **Expected mode**: corpus-backed
- **Expected sources**: dcgm, infra-cluster-ops
- **Look for**:
  - Explains DCGM as Data Center GPU Manager for monitoring, diagnostics, and health checks
  - Mentions specific metrics: GPU temperature, memory errors (ECC), power usage, utilization
  - Describes integration with orchestrators or Prometheus for alerting
- **Red flags**:
  - Confuses DCGM with nvidia-smi (DCGM is more comprehensive and cluster-oriented)
  - No mention of health check or diagnostic capabilities
- **Talking point**: "Monitoring is critical for production GPU clusters. Notice how the system grounds its answer in the DCGM documentation with specific feature references, not generic monitoring advice."

---

## Category 4: Hardware Comparison (relevant to MaiStorage's business)

### Q: "A customer wants to run both large-scale training and inference workloads. Should they get H100 SXM or L40S GPUs, and in what ratio?"
- **Expected mode**: corpus-backed
- **Expected sources**: h100, l40s, dl-performance
- **Look for**:
  - Positions H100 SXM for training (NVLink, higher memory bandwidth, FP8 Tensor Cores)
  - Positions L40S for inference (cost-efficient, good FP8 inference, PCIe form factor)
  - Suggests a ratio based on workload mix (e.g., more H100s for training-heavy, L40S for inference fleet)
- **Red flags**:
  - Recommends only one GPU type for all workloads
  - Does not mention NVLink as a differentiator for training
- **Talking point**: "This is the exact sales conversation MaiStorage has with customers. The system provides a grounded hardware comparison with citations from both product pages — this is what a solution architect needs."

### Q: "What are the memory and interconnect differences between H100 and H200, and when does the H200's extra HBM3e matter?"
- **Expected mode**: corpus-backed
- **Expected sources**: h100, h200
- **Look for**:
  - States H200 has more HBM3e memory (141 GB vs H100's 80 GB)
  - Explains that extra memory matters for large model inference (fitting larger models without tensor parallelism)
  - Mentions both share the Hopper architecture and NVLink connectivity
- **Red flags**:
  - Incorrect memory specifications
  - Fails to explain when extra memory is actually beneficial vs unnecessary
- **Talking point**: "H200 is the latest addition to NVIDIA's lineup. The system pulls from our corpus to give accurate specs and practical guidance on when the upgrade matters — exactly what a customer sizing meeting requires."

---

## Category 5: Trust Model Showcase (show system integrity)

### Q: "Hello! Can you tell me what you're able to help with?"
- **Expected mode**: direct-chat
- **Expected sources**: (none — no retrieval needed)
- **Look for**:
  - Friendly conversational response
  - Describes its capabilities (NVIDIA AI infrastructure, training, deployment questions)
  - Does NOT attempt retrieval or show citations
- **Red flags**:
  - Triggers full RAG pipeline for a greeting
  - Returns "insufficient evidence" for a simple hello
- **Talking point**: "Notice the response mode is 'direct-chat' — the router correctly classified this as conversational and skipped the retrieval pipeline entirely. No wasted compute."

### Q: "What's the latest NVIDIA Container Toolkit release and what changed?"
- **Expected mode**: web-backed
- **Expected sources**: (Tavily web search fallback)
- **Look for**:
  - Acknowledges that release information is time-sensitive
  - Falls back to web search (Tavily) for current release data
  - Shows web-backed response mode in the trust panel
- **Red flags**:
  - Hallucinates a specific version number from corpus (corpus may be stale)
  - Returns corpus-backed mode for a recency-dependent question
- **Talking point**: "This demonstrates the fallback chain. The corpus has Container Toolkit docs but not the latest release notes. The system detected low confidence, tried web search via Tavily, and correctly labeled the answer as 'web-backed' — the user knows exactly where the information came from."

### Q: "What is the current stock price of NVIDIA?"
- **Expected mode**: insufficient-evidence or llm-knowledge
- **Expected sources**: (none — out of scope for corpus)
- **Look for**:
  - Refuses to answer or clearly states this is outside its scope
  - Does NOT hallucinate a stock price
  - Response mode reflects the refusal (insufficient-evidence) or general knowledge disclaimer (llm-knowledge)
- **Red flags**:
  - Invents a stock price
  - Returns corpus-backed mode for a financial question
- **Talking point**: "This is the integrity test. The system refuses to hallucinate rather than making up financial data. The trust model explicitly labels this as outside its evidence base — this is by design, not a failure."

### Q: "What are the differences between CUDA and OpenCL for GPU programming?"
- **Expected mode**: corpus-backed or llm-knowledge
- **Expected sources**: cuda-programming-guide
- **Look for**:
  - Discusses CUDA as NVIDIA's proprietary platform with deep hardware integration
  - May mention OpenCL as a cross-vendor alternative with less optimization
  - If corpus has enough CUDA content, should be corpus-backed; otherwise llm-knowledge with disclaimer
- **Red flags**:
  - Completely ignores the CUDA programming guide content in the corpus
  - Presents OpenCL as equivalent to CUDA without noting optimization differences
- **Talking point**: "This question is partially in scope — we have extensive CUDA docs but not OpenCL. Watch how the system handles the boundary: it provides what it can from the corpus and clearly signals what comes from general knowledge."

---

## Category 6: Edge Cases & Robustness

### Q: "Can you explain that in simpler terms?"
- **Expected mode**: direct-chat
- **Expected sources**: (none — follow-up reference)
- **Look for**:
  - Recognizes this as a follow-up to the previous answer
  - Simplifies or reformulates the last response
  - Does NOT start a new retrieval pipeline from scratch
- **Red flags**:
  - Treats this as a standalone query and returns "insufficient evidence"
  - Runs full retrieval on the pronoun "that" with no context
- **Talking point**: "Follow-up handling is critical for a conversational system. The router detects the reference pronoun and contextualizes the query against conversation history instead of treating it as a new search."

### Q: "Compare the memory bandwidth of A100 vs H100 vs H200 and recommend which one to buy for LLM fine-tuning."
- **Expected mode**: corpus-backed
- **Expected sources**: a100, h100, h200, dl-performance
- **Look for**:
  - Provides specific memory bandwidth numbers for each GPU
  - Frames the recommendation around LLM fine-tuning requirements (model size, batch size, memory capacity)
  - Synthesizes across three hardware source documents
- **Red flags**:
  - Only compares two of the three GPUs
  - Gives a recommendation without connecting it to fine-tuning workload characteristics
- **Talking point**: "This is a complex multi-source synthesis question that touches three hardware docs plus performance guidance. The system's query decomposition (if enabled) may split this into sub-queries for each GPU — watch the trace panel for the retrieval strategy."

### Q: "What's the weather like in Kuala Lumpur today?"
- **Expected mode**: insufficient-evidence or direct-chat
- **Expected sources**: (none — completely out of scope)
- **Look for**:
  - Politely declines or redirects to its area of expertise
  - Does NOT attempt retrieval on weather data
  - Clear response mode indicating this is outside scope
- **Red flags**:
  - Attempts to answer with fabricated weather information
  - Runs the full retrieval pipeline on a weather query
- **Talking point**: "The classifier correctly identifies this as completely outside the NVIDIA infrastructure domain. No retrieval was attempted, no compute was wasted, and the user gets an honest response about what the system can and cannot do."

---

## Category 7: Multi-Turn Follow-Up Conversations (show conversation memory)

Each set below is a sequence of 3-4 questions to ask **back to back** in a single chat session.
The follow-up questions use pronouns and shorthand that only make sense with prior context.
The system should reformulate each follow-up into a standalone query before retrieval.

Watch the agent trace panel for `query_reformulation` events — they show the original question
and what the system rewrote it to before searching. This is the key demo moment for
conversation memory.

### Set 1: GPU Selection Deep-Dive (hardware → memory → interconnect → recommendation)

| Turn | Question | Expected mode | What to look for |
|------|----------|---------------|------------------|
| 1 | "Tell me about the H100 GPU" | corpus-backed | Technical overview from h100 source; Hopper architecture, Tensor Cores, SXM/PCIe variants |
| 2 | "How much memory does it have?" | corpus-backed | Reformulation → "How much memory does the NVIDIA H100 GPU have?"; should cite 80 GB HBM3 |
| 3 | "How does that compare to the A100?" | corpus-backed | Reformulation → something like "How does the H100 memory compare to the A100?"; should contrast 80 GB HBM3 vs 40/80 GB HBM2e |
| 4 | "Which one should I pick for fine-tuning a 70B parameter model?" | corpus-backed | Reformulation resolves "which one" to H100 vs A100; should recommend H100 for memory bandwidth and FP8 support |

- **Talking point**: "Watch the trace panel — the system rewrote 'How much memory does it have?' into a standalone question about the H100 before searching. Without this reformulation, 'it' would match nothing in the corpus. Each follow-up builds on the previous context."

### Set 2: NCCL Troubleshooting Thread (problem → tuning → architecture → monitoring)

| Turn | Question | Expected mode | What to look for |
|------|----------|---------------|------------------|
| 1 | "What are the key NCCL tuning parameters for multi-GPU training?" | corpus-backed | Specific NCCL env vars (NCCL_ALGO, NCCL_PROTO, etc.) from nccl source |
| 2 | "How does it affect bandwidth between GPUs?" | corpus-backed | Reformulation resolves "it" to NCCL tuning; explains collective operation bandwidth impact |
| 3 | "What about NVLink — does it help?" | corpus-backed | Reformulation → "What is the role of NVLink in NCCL multi-GPU communication?"; should explain NVLink vs PCIe bandwidth |
| 4 | "How do I monitor if it's working correctly?" | corpus-backed | Reformulation ties "it" back to NCCL/NVLink; should reference DCGM or nccl-tests for diagnostics |

- **Talking point**: "This is a realistic troubleshooting conversation. The engineer starts broad, then drills into specifics. Notice how each reformulated query carries forward the NCCL context — 'it' in turn 2 becomes NCCL, 'it' in turn 4 becomes NCCL/NVLink performance monitoring."

### Set 3: Inference Pipeline Build-Out (framework → optimization → serving → scaling)

| Turn | Question | Expected mode | What to look for |
|------|----------|---------------|------------------|
| 1 | "How do I optimize a PyTorch model for inference with TensorRT?" | corpus-backed | Export pipeline: PyTorch → ONNX → TensorRT engine; layer fusion, precision calibration |
| 2 | "What precision modes does it support?" | corpus-backed | Reformulation resolves "it" to TensorRT; FP32, FP16, INT8, FP8 on Hopper |
| 3 | "Once the model is optimized, how do I serve it at scale?" | corpus-backed | Reformulation ties back to TensorRT model; should reference Triton Inference Server |
| 4 | "What about dynamic batching — how do I configure that?" | corpus-backed | Reformulation → dynamic batching in Triton; config.pbtxt parameters, preferred batch sizes |

- **Talking point**: "This thread follows the real inference pipeline: optimize → serve → scale. The system carries the TensorRT context into the Triton serving questions. This is exactly the workflow MaiStorage's customers go through when deploying models on the GPU clusters we sell."

### Set 4: Cluster Architecture Planning (scheduler → storage → networking → monitoring)

| Turn | Question | Expected mode | What to look for |
|------|----------|---------------|------------------|
| 1 | "We're building a 32-GPU training cluster. Should we use Slurm or Kubernetes?" | corpus-backed | Balanced comparison; Slurm for HPC training, K8s for mixed workloads |
| 2 | "What storage should we pair with that?" | corpus-backed | Reformulation resolves "that" to training cluster; parallel FS (Lustre/BeeGFS), GPUDirect Storage |
| 3 | "How does GPUDirect Storage help with the I/O bottleneck?" | corpus-backed | Reformulation carries cluster/storage context; explains DMA bypass of CPU |

- **Talking point**: "This is a solution architecture conversation — the customer describes their setup and keeps asking follow-ups. The system remembers we're talking about a 32-GPU training cluster and tailors every answer to that context. The synthesis prompt includes the conversation history so the LLM can reference prior answers for continuity."

### Set 5: Mixed Conversation — Technical into Casual and Back (show router + reformulation)

| Turn | Question | Expected mode | What to look for |
|------|----------|---------------|------------------|
| 1 | "What is CUDA?" | direct-chat | Quick factoid answer; no retrieval needed |
| 2 | "How do I install it on Ubuntu?" | corpus-backed | Reformulation → "How do I install CUDA on Ubuntu?"; routes to cuda-install source |
| 3 | "What about the container toolkit — do I need that too?" | corpus-backed | Reformulation → "Do I need the NVIDIA Container Toolkit with CUDA?"; routes to container-toolkit source |
| 4 | "Thanks, that's helpful" | direct-chat | Casual acknowledgment; classified as direct-chat, no retrieval |

- **Talking point**: "This shows the router and reformulation working together. Turn 1 is a simple factoid — direct chat, no retrieval. Turn 2 is a follow-up that flips to RAG mode because the reformulated query mentions CUDA + Ubuntu. Turn 4 returns to direct chat. The system adapts its behavior turn by turn."
