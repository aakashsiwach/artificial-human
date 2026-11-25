from artificial_human.cognition.relational_memory import RelationalMemoryNetwork


def test_co_retrieved_memories_can_complete_a_later_cue() -> None:
    network = RelationalMemoryNetwork(decay=1.0, retrieval_rate=0.1)
    network.learn(["memory-alpha", "memory-beta"], context_id="owner-a.session-a")

    related = network.activate(["memory-alpha"], context_id="owner-a.session-a")

    assert [item.memory_id for item in related] == ["memory-beta"]
    assert related[0].strength > 0.0
    assert network.snapshot()["retrieval_strengthenings"] == 1


def test_explicit_contradiction_removes_and_blocks_a_positive_association() -> None:
    network = RelationalMemoryNetwork(decay=1.0)
    context = "owner-a.session-a"
    network.learn(["memory-current-a", "memory-current-b"], context_id=context)
    assert network.snapshot()["association_count"] == 1

    related = network.activate(
        ["memory-current-a"],
        context_id=context,
        contradictions=[("memory-current-a", "memory-current-b")],
    )
    network.learn(
        ["memory-current-a", "memory-current-b"],
        context_id=context,
        contradictions=[("memory-current-a", "memory-current-b")],
    )

    assert related == ()
    assert network.snapshot()["association_count"] == 0
    assert network.snapshot()["contradiction_suppressions"] == 1


def test_relational_associations_are_context_scoped_and_decay_locally() -> None:
    network = RelationalMemoryNetwork(decay=0.5, retrieval_rate=0.0)
    network.learn(["memory-alpha", "memory-beta"], context_id="owner-a.session-a")
    before = network.snapshot()["mean_association_strength"]

    network.activate([], context_id="owner-b.session-b")
    assert network.snapshot()["mean_association_strength"] == before
    assert network.activate(["memory-alpha"], context_id="owner-b.session-b") == ()
    assert network.activate(["memory-alpha"], context_id="owner-a.session-a")

    after = network.snapshot()["mean_association_strength"]
    assert after < before


def test_retrieval_strengthening_resists_capacity_interference() -> None:
    baseline = RelationalMemoryNetwork(
        learning_rate=0.08, retrieval_rate=0.0, decay=0.995, max_links=1
    )
    reinforced = RelationalMemoryNetwork(
        learning_rate=0.08, retrieval_rate=0.2, decay=0.995, max_links=1
    )
    context = "owner-a.session-a"

    for network in (baseline, reinforced):
        network.learn(["target-cue", "target-memory"], context_id=context)

    # Recalling the target relation strengthens it only in the adaptive variant.
    for _ in range(5):
        baseline.activate([], context_id=context)
        reinforced.activate(["target-cue"], context_id=context)

    for network in (baseline, reinforced):
        network.learn(["distractor-cue", "distractor-memory"], context_id=context)

    baseline_recall = baseline.activate(["target-cue"], context_id=context)
    reinforced_recall = reinforced.activate(["target-cue"], context_id=context)

    assert baseline_recall == ()
    assert [item.memory_id for item in reinforced_recall] == ["target-memory"]
    assert baseline.snapshot()["association_count"] <= 1
    assert reinforced.snapshot()["association_count"] <= 1
    assert reinforced.snapshot()["evicted_links"] == 0


def test_context_cleanup_removes_relations_without_storing_content() -> None:
    network = RelationalMemoryNetwork()
    network.learn(["opaque-record-1", "opaque-record-2"], context_id="owner-a.session-a")
    network.learn(["opaque-record-3", "opaque-record-4"], context_id="owner-b.session-b")

    snapshot = network.snapshot()
    assert snapshot["retains_memory_content"] is False
    assert snapshot["retains_owner_or_session_ids"] is False
    assert network.forget_context("owner-a.session-a") == 1
    assert network.forget_contexts("owner-b.session-") == 1
    assert network.snapshot()["association_count"] == 0
