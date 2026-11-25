from artificial_human.cognition.associative_memory import AssociativeKnowledgeNetwork


def test_coactivation_strengthens_links_and_spreads_related_facts() -> None:
    network = AssociativeKnowledgeNetwork(learning_rate=0.2, decay=1.0)

    first = network.process(
        "Explain attention, memory, and cognitive architecture together."
    )
    first_state = network.snapshot()

    assert first
    assert first_state["association_count"] > 0
    assert first_state["coactivations"] > 0
    assert first_state["inputs_processed"] == 1

    later = network.process("attention")
    assert any(not item.directly_activated for item in later)
    assert all(item.activation >= 0.0 for item in later)
    assert all(item.fact.fact_id for item in later)


def test_associative_links_decay_on_unmatched_inputs() -> None:
    network = AssociativeKnowledgeNetwork(learning_rate=0.2, decay=0.5)
    network.process("Explain attention, memory, and cognitive architecture together.")
    before = network.snapshot()["mean_association_strength"]

    network.process("qzxv unmatched opaque tokens")
    after = network.snapshot()["mean_association_strength"]

    assert after < before
    assert network.snapshot()["inputs_processed"] == 2


def test_associative_memory_stores_no_raw_query_and_resets_only_dynamic_links() -> None:
    network = AssociativeKnowledgeNetwork()
    secret_like_phrase = "Project codeword glacier orchid"
    network.process(secret_like_phrase)
    snapshot = network.snapshot()

    assert snapshot["retains_user_text"] is False
    assert secret_like_phrase not in str(snapshot)
    assert snapshot["source_fact_count"] > 0

    network.reset()
    reset_state = network.snapshot()
    assert reset_state["association_count"] == 0
    assert reset_state["coactivations"] == 0
    assert reset_state["source_fact_count"] == snapshot["source_fact_count"]
