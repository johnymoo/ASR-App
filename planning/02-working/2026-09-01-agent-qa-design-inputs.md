# Agent Q&A design inputs

Date: 2026-09-01

## Authoritative product source

- GitHub issue: https://github.com/johnymoo/ASR-App/issues/1
- Scope: episode-local questions over the structured summary and complete transcript, with server-side OpenAI-compatible Agent configuration, asynchronous execution, citations, follow-up questions, and no cross-episode context.

## Repository observations

- `scripts/podcast_asr_studio_server.py` is the FastAPI gateway and already mounts persistent podcast and meeting task routers.
- `scripts/podcast_asr_task_api.py` persists task state as atomic JSON files and exposes POST/GET polling endpoints under `/api/podcast-asr`.
- `scripts/publish_podcast_asr_site.py` generates static episode reports and full transcript pages. Transcript chunks already have stable anchors such as `full.html#chunk-001`.
- Each transcription JSON chunk contains `chunk_index`, `start`, `end`, `start_ts`, `end_ts`, and the complete chunk text. The current production chunk duration is about five minutes.
- The generated SRT splits each chunk into sentences and interpolates timestamps by sentence length. These sentence timestamps are useful for navigation but are estimates, not model-provided word-level alignment.
- `podcast_summary.json` contains structured topics, entities, takeaways, and a TLDR. It can improve retrieval but cannot be treated as primary evidence when the answer requires a transcript citation.
- The existing summary client already calls an OpenAI-compatible `/chat/completions` endpoint, but it has fixed defaults and no API-key handling. Q&A needs a shared, configurable client boundary rather than copying the fixed summary behavior.

## Production evidence

- Gateway health on `gb10:8020` returned OK on 2026-09-01.
- Production currently has ten published episode workspaces under `~/podcast`.
- The episode `xiaoyuzhou_6a8bd0f61352af56ff3afff8` is titled `《奥德赛》与全球化的第一次崩塌：海民、青铜时代与今天的世界` and has 23 ASR chunks / about 37,928 transcript characters.
- Its transcript contains at least two book references. The strongest direct recommendation is in chunk 4 (`00:19:40`-`00:24:40`): `千面英雄这本书呢是我非常想推荐大家去读一读`.
- Chunk 1 (`00:04:55`-`00:09:55`) also mentions `1177 B.C.: The Year Civilization Collapsed` as a book the speaker likes. Retrieval must preserve enough surrounding text for the answer model to distinguish a direct recommendation from a general mention.

## Design consequences

1. Citation UI should link to a stable transcript chunk and display an approximate sentence timestamp plus the exact stored text excerpt. MVP must not claim word-level timestamp accuracy.
2. Retrieval should operate on overlapping sentence windows inside each chunk, retain the parent chunk anchor, and provide surrounding context. Whole-chunk retrieval is too coarse for citations.
3. A summary may participate in retrieval and query expansion, but final factual answers must cite transcript windows when transcript evidence exists.
4. Question tasks can reuse the existing persistent JSON + polling pattern, but question/session state needs its own namespace and must be scoped by episode ID.
5. Agent configuration must have a dedicated web page. The service still exposes one active Agent profile, using an OpenAI-compatible API.

## User decisions added after initial discovery

Source: user response on 2026-09-01.

- Use one active Agent, backed by an OpenAI-compatible API.
- Provide a web configuration page for that Agent.
- Support follow-up questions in the same conversation.
- Let users switch the question scope in the page.
- Support both one selected episode and the complete podcast library as question scopes.
- Inject structured summary, official outline, and transcript evidence into retrieval/context.
- This explicitly supersedes Issue #1's original non-goal that excluded cross-episode global Q&A.

## Remaining product decision

The existing LAN site has no authentication. A configuration page that can change the endpoint, model, prompt, and API key therefore needs an explicit access policy. The design must decide whether configuration is administrator-protected or editable by every site visitor.

Resolved by the user on 2026-09-01: this is a single-user deployment. The MVP configuration page does not require authentication. The saved API key must still be masked in responses and must never be emitted into static pages or logs.
