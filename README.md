# Skill Data Synthesis Pipeline

自动化 pipeline，从开源 skill/prompt 集合中采集数据，利用 LLM 合成高质量训练数据，输出 JSONL 格式用于模型微调。

## 数据流

```
GitHub repos / CSV URLs
        │
        ▼
   [Collectors]  ── async HTTP ──> raw text/CSV/JSON
        │
        ▼
   list[Skill]   (~2500 raw skills, content hash 去重)
        │
        ▼
   [Heuristic Filter]  ── 长度/语言/黑名单
        │
        ▼
   [Synthesizer]  ── 4 strategies × N samples/skill ── LLM API (async + semaphore)
        │
        ▼
   [Sample Filter]  ── 长度/去重/refusal 检测
        │
        ▼
   [LLM Quality Filter]  ── 抽样评分
        │
        ▼
   training_data.jsonl
```

## 快速开始

```bash
# 创建虚拟环境并安装
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 或 ANTHROPIC_API_KEY

# 试运行（仅采集 + 过滤，不调用 LLM）
skill-synth --dry-run

# 小规模测试（5 个 skill）
skill-synth --limit 5

# 全量运行
skill-synth
```

## 数据源

| 来源 | 类型 | 预期数量 |
|------|------|----------|
| [awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts) | CSV | ~1200 |
| [TheBigPromptLibrary](https://github.com/0xeb/TheBigPromptLibrary) | Markdown | ~110 |
| [System-Prompt-Library](https://github.com/danielrosehill/System-Prompt-Library) | JSON | ~1260 |

可通过 `config.yaml` 的 `sources.github_repos` 添加更多仓库。

## 合成策略

每个 skill 使用 4 种策略生成训练样本：

| 策略 | 说明 | 默认数量/skill |
|------|------|----------------|
| **Direct Task** | 生成多样化用户查询 + 专家回答 | 5 |
| **Scenario-Based** | 构造具体场景后生成 Q&A | 3 |
| **Conversation** | 多轮对话，提取独立 instruction-response 对 | 4 |
| **Edge Case** | 模糊请求/超范围/误解纠正/高难度 | 3 |

## 配置

所有配置集中在 `config.yaml`：

```yaml
llm:
  provider: openai        # openai | anthropic
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY

synthesis:
  concurrency: 10         # 并发 LLM 请求数
  temperature: 0.8

filters:
  llm_quality:
    enabled: true
    sample_ratio: 0.2     # 抽样 20% 做质量评估
    min_overall_score: 3.5
```

## 输出格式

`data/output/training_data.jsonl`，每行一个 JSON 对象：

```json
{
  "instruction": "用户指令",
  "response": "模型回答",
  "system_prompt": "",
  "skill_id": "xxhash hex",
  "strategy": "direct_task",
  "difficulty": "medium",
  "quality_score": 4.2,
  "metadata": {"skill_name": "...", "scenario": "..."}
}
```

## 成本预估

使用 `gpt-4o-mini` 全量运行（~2500 skills × 15 samples/skill）约 $3.50。
