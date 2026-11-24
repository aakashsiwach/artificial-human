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

    provider = RecordingProvider()
    brain = BrainService(provider, LearningSystem(repository))
    response = brain.respond(
        owner_id="owner",
        query="How should I explain a backend concept?",
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
