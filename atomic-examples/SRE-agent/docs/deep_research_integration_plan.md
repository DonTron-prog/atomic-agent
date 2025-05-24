# Deep Research Integration Plan for SRE Agent

This document outlines an incremental approach to integrate the `deep-research` functionality into the SRE agent, focusing on modularity and adherence to Atomic Design principles.

## User Requirements

The primary goals for this integration are:
1.  Enable webpage scraping and content extraction capabilities.
2.  Incorporate follow-up question generation to facilitate deeper investigation.
3.  Implement persistent memory for gathered information, with the eventual aim of making this memory accessible to other agents within the SRE system.

## Incremental Integration Phases

### Phase 1: Integrate Webpage Scraping as a Standalone Tool

*   **Goal**: Enable the SRE Orchestrator to directly use a webpage scraping tool for immediate content retrieval from URLs.
*   **Rationale**: This is a foundational capability, provides quick value, and is a prerequisite for more advanced research functions.
*   **Actions**:
    1.  **Adapt `WebpageScraperTool`**:
        *   Copy the `WebpageScraperTool` class, its input schema (`WebpageScraperToolInputSchema`), output schema (`WebpageScraperToolOutputSchema`), and config (`WebpageScraperToolConfig`) from `atomic-examples/deep-research/deep_research/tools/webpage_scraper.py` into the SRE agent's tool directory (e.g., `atomic-examples/SRE-agent/orchestration_agent/tools/`).
        *   Ensure it's self-contained or that its dependencies are met within the SRE agent's environment.
    2.  **Update SRE Orchestrator (`atomic-examples/SRE-agent/orchestration_agent/orchestrator.py`)**:
        *   Import the `WebpageScraperTool` and its schemas.
        *   Add `WebpageScraperToolInputSchema` to the `Union` in `OrchestratorOutputSchema.tool_parameters`.
        *   Update the `tool` field description in `OrchestratorOutputSchema` to include `'webpage_scraper'`.
        *   Modify the `initialize_tools` function to instantiate and include the `WebpageScraperTool`.
        *   Extend the `execute_tool` function to handle the `webpage_scraper` tool.
        *   Update the orchestrator agent's system prompt (within `create_orchestrator_agent` function) to instruct it when to use the `webpage_scraper` tool (e.g., "If an alert or context provides a direct URL to documentation, an error log, or a status page, use the 'webpage_scraper' tool to fetch its content.").
*   **Outcome**: The SRE agent can now be directed by the orchestrator to scrape specific URLs.

### Phase 2: Introduce a `DeepResearchTool` for Comprehensive Research & Follow-up Questions

*   **Goal**: Create a new tool for the SRE Orchestrator that encapsulates a multi-step research process (query generation, search, scraping, and QA with follow-up questions).
*   **Rationale**: This directly addresses the need for more comprehensive research and the generation of follow-up questions.
*   **Actions**:
    1.  **Create `DeepResearchTool` Components in SRE Agent**:
        *   In `atomic-examples/SRE-agent/orchestration_agent/tools/`, create a new file, e.g., `deep_research_tool.py`.
        *   Define `DeepResearchToolInputSchema`:
            *   `research_topic: str`
            *   `num_queries_to_generate: int` (optional, default to 3)
            *   `num_pages_to_scrape: int` (optional, default to 3)
        *   Define `DeepResearchToolOutputSchema`:
            *   `summary: str`
            *   `sources: List[HttpUrl]`
            *   `follow_up_questions: List[str]`
        *   Implement the `DeepResearchTool(BaseTool)` class:
            *   Its `run` method will orchestrate:
                *   **Query Generation**: Adapt `QueryAgent` logic from `deep-research`.
                *   **Web Search**: Use the SRE agent's existing `SearxNGSearchTool`.
                *   **Webpage Scraping**: Use the `WebpageScraperTool` (from Phase 1).
                *   **Content Aggregation & Context Provision**: Adapt `ScrapedContentContextProvider`.
                *   **Question Answering & Follow-ups**: Adapt `QuestionAnsweringAgent` logic, registering the `ScrapedContentContextProvider`.
    2.  **Update SRE Orchestrator (`atomic-examples/SRE-agent/orchestration_agent/orchestrator.py`)**:
        *   Import the `DeepResearchTool` and its schemas.
        *   Add `DeepResearchToolInputSchema` to the `Union` in `OrchestratorOutputSchema.tool_parameters`.
        *   Update the `tool` field description to include `'deep_research'`.
        *   Modify `initialize_tools` to instantiate and include `DeepResearchTool`.
        *   Extend `execute_tool` to handle the `deep_research` tool.
        *   Update the orchestrator agent's system prompt for when to use the `deep_research` tool.
*   **Outcome**: The SRE agent can perform in-depth research, providing a summary, sources, and follow-up questions.

### Phase 3: Implement Persistent Memory within `DeepResearchTool`

*   **Goal**: Enable the `DeepResearchTool` to store and retrieve scraped content and research summaries, making its own operations more efficient.
*   **Rationale**: Addresses persistent memory, initially scoped to the `DeepResearchTool`.
*   **Actions**:
    1.  **Integrate `ChromaDBService` into `DeepResearchTool`**:
        *   The `DeepResearchTool` will use an instance of `ChromaDBService` from `atomic-examples/SRE-agent/orchestration_agent/services/chroma_db.py`.
        *   Use a dedicated collection name (e.g., `sre_deep_research_cache`).
    2.  **Modify `DeepResearchTool.run()` Logic**:
        *   **Cache Check**: Before generating queries, query its ChromaDB collection. If relevant, recent info is found, potentially return cached summary/follow-ups or augment new search.
        *   **Cache Update**: After successful scraping/summary, store raw scraped content, summary, and follow-ups in ChromaDB with metadata (URL, timestamp, topic).
    3.  **Configuration**:
        *   Ensure `DeepResearchToolConfig` includes parameters for its ChromaDB collection name.
*   **Outcome**: `DeepResearchTool` becomes more efficient by reusing previously processed information.

### Phase 4: Externalize Persistent Memory for Broader SRE Agent Access

*   **Goal**: Make knowledge gathered by `DeepResearchTool` (and potentially other tools) accessible to the entire SRE agent system.
*   **Rationale**: Fulfills the vision of a shared, persistent knowledge base.
*   **Actions**:
    1.  **Unified Knowledge Store Design**:
        *   Evaluate expanding the existing `ChromaDBService` or create a new, dedicated `KnowledgeStoreService` (possibly still using ChromaDB) with a generalized schema.
    2.  **Standardized Knowledge Schema**:
        *   Define a common schema for documents in this unified store (content, source_url, source_type, timestamp, original_query_or_alert_id, related_entities, summary, follow_up_questions).
    3.  **Modify `DeepResearchTool`**:
        *   Writes findings to this unified knowledge store.
        *   Queries this unified store at the beginning of its process.
    4.  **Update SRE Orchestrator & Other Potential Agents**:
        *   The `OrchestratorAgent` (or a future Planner/Reflection Agent) can query this unified store *before* deciding on a tool.
        *   The RAG tool might also query this broader set of documents.
    5.  **New Context Providers**:
        *   Create context providers that can fetch and format information from this unified knowledge store for various agents.
*   **Outcome**: A shared, persistent knowledge base improves overall SRE agent decision-making and planning.