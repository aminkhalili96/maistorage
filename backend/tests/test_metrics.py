from app.services.evaluation import hit_at_k, ndcg, reciprocal_rank


def test_hit_at_k_detects_relevant_source():
    assert hit_at_k(["nccl", "cuda-programming-guide"], ["nccl"]) == 1.0


def test_reciprocal_rank_rewards_early_match():
    assert reciprocal_rank(["nccl", "fabric-manager"], ["fabric-manager"]) == 0.5


def test_ndcg_is_zero_without_relevant_results():
    assert ndcg(["cuda-install", "container-toolkit"], ["nccl"]) == 0.0


def test_ndcg_deduplicates_repeated_source_hits():
    assert ndcg(["nccl", "nccl", "fabric-manager"], ["nccl", "fabric-manager"]) <= 1.0
