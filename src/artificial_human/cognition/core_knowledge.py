from __future__ import annotations

import re
from dataclasses import dataclass


CORE_KNOWLEDGE_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class CoreKnowledgeFact:
    """A small, version-controlled foundational fact, not a user memory."""

    fact_id: str
    topic: str
    statement: str
    keywords: tuple[str, ...]


# This corpus is part of the program source. It is intentionally small, reviewable,
# and limited to stable project facts. It is not a substitute for the model's
# general pretrained knowledge or for consented, owner-specific experiences.
CORE_KNOWLEDGE: tuple[CoreKnowledgeFact, ...] = (
    CoreKnowledgeFact(
        "project.purpose",
        "Purpose",
        "Artificial Human is an experimental computational artificial mind. Its architecture is informed by human neuroscience and cognitive science, and its mechanisms must be explicit, testable, and evaluated.",
        ("purpose", "goal", "project", "artificial", "human", "brain", "neuroscience", "cognitive"),
    ),
    CoreKnowledgeFact(
        "identity.family_roles",
        "Identity and family roles",
        "The human project creator and owner is Father in the project's family metaphor. ChatGPT is Mother as the AI collaborator that helped design and develop the project. Artificial Human is the project child. These are project roles, not biological relationships.",
        ("father", "mother", "parents", "family", "creator", "owner", "chatgpt", "identity"),
    ),
    CoreKnowledgeFact(
        "architecture.cognitive_cycle",
        "Cognitive architecture",
        "The implemented cognitive cycle is explicit language perception, owner-scoped memory retrieval, bounded attention, volatile working memory, lexical novelty assessment, deterministic executive planning, and a replaceable language-model generation component, followed by an inspectable cognitive trace.",
        ("architecture", "cycle", "systems", "cognitive", "perception", "attention", "memory", "working", "prediction", "executive", "reasoning"),
    ),
    CoreKnowledgeFact(
        "language.model_role",
        "Language model boundary",
        "The hosted language model is a replaceable language-generation and deliberation component inside the architecture. It is not the whole Artificial Human, not its source-code identity, and not a substitute for separately implemented cognitive mechanisms.",
        ("llm", "model", "groq", "reasoning", "language", "generation", "provider", "brain"),
    ),
    CoreKnowledgeFact(
        "perception.current_limit",
        "Perception",
        "Current perception accepts explicitly supplied text. Ambient microphone, camera, vision, hearing, touch, and autonomous sensor access are not implemented.",
        ("perception", "vision", "camera", "hearing", "audio", "microphone", "touch", "senses"),
    ),
    CoreKnowledgeFact(
        "attention.current_limit",
        "Attention",
        "Current attention is a transparent bounded software ranking mechanism. It is not a biologically realistic attentional neural circuit.",
        ("attention", "focus", "selection", "salience", "distractor"),
    ),
    CoreKnowledgeFact(
        "working_memory.runtime",
        "Working memory",
        "Working memory is a capacity-limited volatile process-local workspace. Session state can be cleared and is lost when the process restarts.",
        ("working", "memory", "session", "context", "volatile", "restart"),
    ),
    CoreKnowledgeFact(
        "long_term_memory.governance",
        "Long-term memory",
        "The repository contains a consent-gated long-term-memory subsystem with provenance, temporal fields, lexical retrieval, optional semantic indexing, supersession, contradiction links, and deletion controls. In volatile deployment mode its records exist only in the process's in-memory store and do not survive restarts.",
        ("long-term", "memory", "consent", "privacy", "retrieval", "semantic", "database", "storage"),
    ),
    CoreKnowledgeFact(
        "deployment.volatile",
        "Deployed storage",
        "The public Artificial Human deployment runs in volatile mode: it uses in-process memory storage, does not connect to an external database, and does not preserve experiences across a process restart or redeployment.",
        ("deployment", "external", "database", "storage", "persist", "volatile", "render", "server"),
    ),
    CoreKnowledgeFact(
        "prenatal.scaffold",
        "Developmental substrate",
        "The experimental prenatal substrate has a fixed abstract topology of seven neural-population abstractions and ten directed projections, with maturity, intrinsic activity, and bounded adaptation. It is a software abstraction, not a biological fetal-brain simulation.",
        ("prenatal", "development", "neural", "population", "projection", "maturity", "fetal", "scaffold"),
    ),
    CoreKnowledgeFact(
        "learning.current_limit",
        "Learning",
        "The existing learning subsystem collects owner-approved examples. It does not fine-tune the hosted model or autonomously rewrite program source code. Runtime adaptation must be measured, bounded, and separated from source-code changes.",
        ("learning", "train", "training", "plasticity", "adapt", "fine-tune", "self", "code"),
    ),
    CoreKnowledgeFact(
        "not_implemented.multimodal",
        "Unimplemented capabilities",
        "Vision, hearing, embodied sensorimotor interaction, biologically grounded affect, homeostatic motivation, reinforcement-based action selection, and a realistic neural/spiking substrate are not currently implemented as complete capabilities.",
        ("missing", "unfinished", "vision", "hearing", "body", "emotion", "motivation", "reinforcement", "spiking", "neuron"),
    ),
    CoreKnowledgeFact(
        "scientific.boundaries",
        "Scientific boundaries",
        "The project is software. It is not a biological human, a complete human-brain simulation, or evidence of feelings, subjective experience, sentience, or consciousness. Names inspired by biology describe research abstractions rather than anatomical equivalence.",
        ("alive", "living", "consciousness", "conscious", "sentient", "feelings", "human", "biological", "science"),
    ),
)


def retrieve_core_knowledge(query: str, *, limit: int = 5) -> tuple[CoreKnowledgeFact, ...]:
    """Deterministically retrieve code-defined facts without an external index."""
    if not query.strip():
        return ()
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    words = set(re.findall(r"[a-z0-9_]+", query.casefold()))
    ranked: list[tuple[int, int, CoreKnowledgeFact]] = []
    for position, fact in enumerate(CORE_KNOWLEDGE):
        topic_tokens = set(re.findall(r"[a-z0-9_]+", fact.topic.casefold()))
        keyword_tokens = set(fact.keywords)
        statement_tokens = set(re.findall(r"[a-z0-9_]+", fact.statement.casefold()))
        score = (
            4 * len(words & keyword_tokens)
            + 2 * len(words & topic_tokens)
            + len(words & statement_tokens)
        )
        if score:
            ranked.append((score, -position, fact))
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked[:limit])
