# Module 3: Support Assistant RAG Service (`/support_assistant`)

This module implements a complete, local, zero-cost Retrieval-Augmented Generation (RAG) service for Zepto's policy management. It incorporates document embedding, ChromaDB vector indexing, a LangGraph state graph with conditional intent routing, Pydantic JSON schema validation, and a FastAPI endpoint wrapped in a Docker container.

---

## 1. RAG Pipeline Architecture

```text
[User Query]
     │
     ▼
┌─────────────────────────┐
│  classify_intent_node   │ ──(Keyword Heuristic)
└─────────────────────────┘
     │
     ├──► policy_question ──► ┌──────────────────────────┐
     │                        │ retrieve_and_answer_node │
     │                        └──────────────────────────┘
     │                                     │
     │                         (ChromaDB Cosine Search)
     │                                     │
     │                        ┌──────────────────────────┐
     │                        │  MOCK_LLM / Real LLM     │
     │                        └──────────────────────────┘
     │                                     │
     └──► general_question──► ┌──────────────────────────┐
                              │    direct_answer_node    │
                              └──────────────────────────┘
                                           │
                                           ▼
                                [Pydantic JSON Response]