from artificial_human.brain.providers.base import ChatMessage, Generation
from artificial_human.brain.service import BrainService
from artificial_human.core.models import MemoryKind
from artificial_human.identity.profile import IdentityProfile
from artificial_human.learning.system import LearningSystem
from artificial_human.memory.repository import MemoryRepository


class RecordingProvider:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def generate(
        self, messages: list[ChatMessage], *, temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Generation:
        self.messages = messages
        return Generation(
            content="You prefer concise technical explanations.",
            model="test-model",
            prompt_tokens=40,
            completion_tokens=9,
        )


def test_brain_uses_profile_recalled_memory_and_approved_demonstrations(tmp_path) -> None:
    repository = MemoryRepository(tmp_path / "brain.sqlite3")
    memory_consent = repository.grant_consent("owner", purpose="personal_memory")
    repository.add(
        owner_id="owner",
        kind=MemoryKind.PREFERENCE,
        content="I prefer concise technical explanations with practical examples.",
        source="owner statement",
        consent_id=memory_consent.id,
    )
    learning_consent = repository.grant_consent("owner", purpose="personal_learning")
    example = repository.add_training_example(
        owner_id="owner",
        task_category="communication",
        input_text="Explain a backend concept",
        target_text="Use a direct explanation, then give a practical example.",
        source="owner-approved demonstration",
        consent_id=learning_consent.id,
    )
    assert repository.review_training_example(
        "owner", example.id, decision="approved", reviewer="owner"
    )

    approved_examples = LearningSystem(repository).examples(
        "owner", status="approved"
    )
    assert [item.id for item in approved_examples] == [example.id]
    ranked_examples = BrainService.retriever_rank_examples(
        "How should I explain a backend concept with concise technical explanations?",
        approved_examples,
        limit=3,
    )
    assert [item.id for item, _ in ranked_examples] == [example.id]

    provider = RecordingProvider()
    brain = BrainService(provider, LearningSystem(repository))
    response = brain.respond(
        owner_id="owner",
        query="How should I explain a backend concept with concise technical explanations?",
        profile=IdentityProfile(
            owner_id="owner",
            display_name="Owner",
            communication_preferences={"tone": "natural"},
        ),
        memories=repository.list("owner"),
        mode="style_clone",
    )

    assert response.model == "test-model"
    assert response.mode == "style_clone"
    assert response.recalled_memories[0]["source"] == "owner statement"
    assert response.demonstrations[0]["example_id"] == example.id
    assert "OWNER-AUTHORED IDENTITY PROFILE" in provider.messages[0].content
    assert "RETRIEVED MEMORY EVIDENCE" in provider.messages[1].content
    assert "OWNER-APPROVED STYLE / TASK EXAMPLES" in provider.messages[1].content
    assert response.prompt_tokens == 40
    assert response.cognitive_trace["perception"]["modality"] == "language"
    assert response.cognitive_trace["executive_control"]["approval_required"] is True
    assert "BOUNDED WORKING-MEMORY CONTEXT" in provider.messages[1].content
    assert response.session_id is None


def test_brain_includes_relevant_source_code_knowledge_in_cognitive_context() -> None:
    repository = MemoryRepository(":memory:")
    provider = RecordingProvider()
    brain = BrainService(provider, LearningSystem(repository))

    response = brain.respond(
        owner_id="owner",
        query="What cognitive systems are implemented in the architecture?",
        profile=IdentityProfile(owner_id="owner", display_name="Artificial Human"),
    )

    assert response.model == "test-model"
    assert "TRUSTED SOURCE-CODE FOUNDATIONAL KNOWLEDGE" in provider.messages[1].content
    assert "cognitive_cycle" in provider.messages[1].content
    assert "version_controlled_source_code" in provider.messages[1].content


def test_brain_runs_recurrent_dynamics_and_includes_state_in_context() -> None:
    repository = MemoryRepository(":memory:")
    provider = RecordingProvider()
    brain = BrainService(provider, LearningSystem(repository))

    first = brain.respond(
        owner_id="owner",
        query="How does episodic memory work and why?",
        profile=IdentityProfile(owner_id="owner"),
    )
    dynamics = first.cognitive_trace["recurrent_dynamics"]

    assert dynamics["version"] == "recurrent-population-dynamics-v1"
    assert dynamics["inputs_processed"] == 1
    assert dynamics["integration_steps"] == 8
    assert dynamics["population_count"] == 7
    assert dynamics["projection_count"] == 10
    assert dynamics["plasticity_enabled"] is False
    assert dynamics["state_storage"] == "volatile_process_memory"
    assert dynamics["active_population_count"] > 0
    assert "RECURRENT ABSTRACT POPULATION ACTIVITY" in provider.messages[1].content

    second = brain.respond(
        owner_id="owner",
        query="Plan the next implementation steps safely.",
        profile=IdentityProfile(owner_id="owner"),
    )
    assert second.cognitive_trace["recurrent_dynamics"]["inputs_processed"] == 2
    assert second.cognitive_trace["recurrent_dynamics"]["integration_steps"] == 16


def test_brain_dynamics_state_can_be_reset() -> None:
    repository = MemoryRepository(":memory:")
    brain = BrainService(RecordingProvider(), LearningSystem(repository))
    brain.dynamics.process("Explain recurrent activity.")
    brain.dynamics.reset()
    snapshot = brain.dynamics.snapshot().as_dict()

    assert snapshot["inputs_processed"] == 0
    assert snapshot["integration_steps"] == 0
    assert snapshot["mean_activity"] == 0.0
    assert all(item["activity"] == 0.0 for item in snapshot["populations"])
