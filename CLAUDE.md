# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate && pip install -e .

# Run pipeline
skill-synth                          # full pipeline (requires API key in .env)
skill-synth --dry-run                # collect + filter only, no LLM calls
skill-synth --limit 5                # process only 5 skills
skill-synth --config path/to.yaml    # custom config file

# Environment: requires .env with OPENAI_API_KEY or ANTHROPIC_API_KEY
# Note: GITHUB_TOKEN is optional but increases GitHub API rate limits
```

## Architecture

This is a 6-stage async pipeline that collects LLM skill/prompts from the web, uses an LLM to synthesize training data (instruction-response pairs), and outputs JSONL.

**Data flow:** `Collectors → Skill Filter → Synthesizer → Sample Filter → LLM Quality Filter → JSONL Writer`

### Pipeline stages (orchestrated by `pipeline.py`)

1. **Collect** (`collectors/`): Async HTTP fetchers run concurrently, deduplicate results by `Skill.id` (xxhash of content). Two collector types: `AwesomePromptsCollector` (CSV), `GitHubRepoCollector` (Trees API + raw content fetch, supports markdown/json/plaintext parsers).
2. **Skill Filter** (`filters/heuristic.py:HeuristicFilter`): Length, word count, ASCII ratio, blacklist patterns.
3. **Synthesize** (`synthesis/`): 4 strategies (direct_task, scenario_based, conversation, edge_case) defined as prompt templates in `prompts.py`. `Synthesizer` dispatches all skill×strategy combinations concurrently via `asyncio.Semaphore`. LLM returns JSON; parsed into `TrainingSample` objects.
4. **Sample Filter** (`filters/heuristic.py:SampleHeuristicFilter`): Instruction/response length, dedup by normalized instruction, refusal pattern detection.
5. **LLM Quality** (`filters/llm_filter.py`): Scores a random subset (default 20%) on 5 dimensions (1-5 scale). Removes samples below threshold.
6. **Write** (`output/writer.py`): JSONL + stats JSON.

### LLM abstraction (`llm/`)

`BaseLLMClient` with `generate()` and `generate_json()` methods. Factory `create_llm_client()` in `__init__.py` switches between `OpenAIClient` (uses `response_format=json_object`) and `AnthropicClient` based on `config.llm.provider`.

### Configuration

All config flows through `config.py:AppConfig` (Pydantic models). Loaded from `config.yaml` + `.env`. The `config.yaml` controls sources, filter thresholds, synthesis strategies/concurrency, and output paths. LLM API keys are read from env vars specified by `llm.api_key_env`.

### Key data models (`models.py`)

- `Skill`: collected prompt. `id` is a computed field (xxhash of normalized content).
- `TrainingSample`: instruction/response pair with strategy, difficulty, quality_score.
- `QualityVerdict`: 5-dimension scores with computed `overall` average.
