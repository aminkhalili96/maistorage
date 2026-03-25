# Speculative Decoding for LLM Inference Optimization

Speculative decoding accelerates autoregressive LLM inference by using a smaller, faster draft model to propose multiple tokens which are then verified in parallel by the target model. This achieves 2-3x speedup in token generation without any change to the output distribution -- the quality of generated text remains mathematically identical to standard autoregressive decoding.

## The Autoregressive Bottleneck

Standard LLM inference generates tokens one at a time. Each token requires a full forward pass through the model, reading every parameter from GPU HBM (High Bandwidth Memory) to compute a single output. For large models (70B+ parameters), this creates a severe memory-bandwidth bottleneck.

During decoding, arithmetic intensity (compute operations per memory read) is extremely low. The GPU reads hundreds of gigabytes of weights from HBM but computes very little per parameter per token. GPU compute utilization during autoregressive decoding typically ranges from 5% to 15%, even on H100 or A100 hardware. Tensor cores sit mostly idle waiting for weight data from memory.

This bottleneck is fundamental: each token depends on all previous tokens through the KV cache, creating a strict sequential dependency that prevents naive parallelization.

## How Speculative Decoding Works

Speculative decoding introduces a two-phase generate-then-verify approach:

**Draft phase:** A small draft model (1-7B parameters) generates K candidate tokens autoregressively. Because the draft model is much smaller, each forward pass is fast and K tokens are generated quickly.

**Verification phase:** The target model (the large model whose quality we want) processes the prefix plus all K draft tokens in a single forward pass. Transformer attention processes multiple tokens in parallel (similar to prefill), so this single pass costs roughly the same as generating one token but verifies K tokens at once.

**Acceptance/rejection:** Each draft token is accepted or rejected using modified rejection sampling. The algorithm compares draft and target model token probabilities. Agreeing tokens are accepted; disagreements are probabilistically rejected. This guarantees the output follows the exact same distribution as standard decoding from the target model -- no quality loss.

Expected acceptance rates range from 70% to 90% depending on draft model quality. When a token is rejected, a corrected token is sampled from an adjusted distribution.

## Draft Model Selection

**Same-family smaller model:** Pair models from the same family (e.g., Llama 3.1 70B target with Llama 3.1 8B draft). Good acceptance rates because same-family models share training data and patterns.

**Distilled or pruned model:** A version of the target distilled to be smaller. Often achieves higher acceptance rates than generic small models since it specifically matches the target's distribution.

**Self-speculative decoding:** Use early-exit layers from the target model itself as the draft predictor. Avoids needing a separate model but requires architectural modifications.

**Medusa:** Adds multiple parallel prediction heads to the target model, each predicting a future token at a different position. No separate draft model needed. Verification uses tree-based attention to check all predictions in parallel.

## TensorRT-LLM Support

NVIDIA's TensorRT-LLM implements speculative decoding as a first-class feature:

- **Draft model approach:** Specify a separate draft engine alongside the target engine. TensorRT-LLM handles KV cache management, scheduling, and acceptance/rejection automatically.
- **Medusa approach:** Build the target model with Medusa heads compiled into the TensorRT engine with tree-based verification attention.
- **Automatic tuning:** TensorRT-LLM selects the number of speculative tokens (K) based on available GPU memory and observed acceptance rates, adapting at runtime.

Configuration uses `executor_config.speculative_config` to set draft model path, draft token count, and acceptance thresholds.

## Performance Characteristics

Speedup depends on:
- **Acceptance rate:** 80-90% acceptance yields 2-3x speedups.
- **Draft model speed:** Should be small compared to target model's forward pass. A 10x smaller draft typically suffices.
- **Speculated tokens (K):** More tokens mean higher potential speedup but diminishing returns. Typical values: 3-8.

Observed speedups:
- **Code generation and structured text:** 2-3x (high acceptance due to predictable patterns)
- **Creative and diverse text:** 1.5-2x (more varied distributions lower acceptance)
- **Mathematical reasoning:** 2-2.5x (structured output with predictable formatting)

No quality degradation -- rejection sampling mathematically guarantees identical output distribution.

## Trade-offs

**Memory overhead:** The draft model requires additional GPU memory. A 7B draft adds approximately 14 GB HBM in FP16, which must be available alongside the target model and both KV caches.

**Batch complexity:** Each request may accept different numbers of draft tokens, creating variable-length sequences. The inference engine must handle this heterogeneity efficiently.

**Diminishing returns at high batch sizes:** Speculative decoding addresses the memory-bandwidth bottleneck most severe at low batch sizes. At high batch sizes, arithmetic intensity increases naturally and GPU utilization improves without speculation.

**Optimal scenarios:** Most effective for interactive, low-batch inference (chatbots, coding assistants, real-time applications) where per-token latency matters. For high-throughput batch inference, continuous batching with large batch sizes may achieve comparable GPU utilization without speculation complexity.
