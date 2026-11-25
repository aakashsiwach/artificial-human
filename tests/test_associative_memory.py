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



def test_associations_are_scoped_to_context_and_keep_provenance() -> None:
    network = AssociativeKnowledgeNetwork(learning_rate=0.2, decay=0.5)

    network.process("purpose mother", context_id="conversation-a")
    before_other_context = network.snapshot()["mean_association_strength"]

    # Activity in another context must not decay the first context's links.
    network.process("unmatched opaque tokens", context_id="conversation-b")
    assert network.snapshot()["mean_association_strength"] == before_other_context

    other_context = network.process("purpose", context_id="conversation-b")
    assert not any(item.fact.fact_id == "identity.family_roles" for item in other_context)

    same_context = network.process("purpose", context_id="conversation-a")
    target = next(item for item in same_context if item.fact.fact_id == "identity.family_roles")
    assert target.directly_activated is False
    assert target.association_context_id == "conversation-a"
    assert network.snapshot()["association_context_count"] == 1


def test_association_context_identifiers_reject_unsafe_values() -> None:
    network = AssociativeKnowledgeNetwork()
    for context_id in ("", "session id", "../../private", "x" * 65, "UPPERCASE"):
        try:
            network.process("purpose mother", context_id=context_id)
        except ValueError as exc:
            assert "context_id" in str(exc)
        else:
            raise AssertionError(f"unsafe context identifier was accepted: {context_id!r}")


def test_forget_context_and_prefix_remove_only_matching_associations() -> None:
    network = AssociativeKnowledgeNetwork()
    network.process("purpose mother", context_id="owner-a.session-one")
    network.process("purpose mother", context_id="owner-a.session-two")
    network.process("purpose mother", context_id="owner-b.session-three")

    assert network.snapshot()["association_context_count"] == 3
    assert network.forget_context("owner-a.session-one") is True
    assert network.forget_context("owner-a.session-one") is False
    assert network.forget_contexts("owner-a.session-") == 1

    snapshot = network.snapshot()
    assert snapshot["association_context_count"] == 1
    assert snapshot["association_count"] > 0

    assert network.forget_contexts("owner-b.session-") > 0
    assert network.snapshot()["association_count"] == 0
