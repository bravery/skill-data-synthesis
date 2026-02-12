"""Synthesis prompt templates for the 4 generation strategies."""

DIRECT_TASK_PROMPT = """\
You are generating diverse, high-quality training data for an AI assistant.

Given the following expert skill/prompt, generate {n} diverse user queries that someone would ask an expert with this skill, along with detailed expert responses.

Skill/Prompt:
---
{skill_content}
---

Requirements:
- Each query should be realistic and specific
- Responses should demonstrate deep expertise from the skill
- Vary the difficulty: include easy, medium, and hard queries
- Responses should be detailed, actionable, and educational
- Do NOT generate meta-commentary about the skill itself

Respond with a JSON object:
{{"samples": [
  {{"instruction": "user query", "response": "expert response", "difficulty": "easy|medium|hard"}},
  ...
]}}"""

SCENARIO_BASED_PROMPT = """\
You are generating scenario-based training data for an AI assistant.

Given the following expert skill/prompt, first create {n} concrete, realistic scenarios where this expertise would be needed. Then generate a user question and expert answer for each scenario.

Skill/Prompt:
---
{skill_content}
---

Requirements:
- Each scenario should be a specific, realistic situation (e.g., a startup founder, a student, a team lead)
- The question should naturally arise from the scenario context
- The response should be tailored to the specific scenario
- Include practical, actionable advice

Respond with a JSON object:
{{"samples": [
  {{"scenario": "brief scenario description", "instruction": "user question in context", "response": "expert response tailored to scenario", "difficulty": "easy|medium|hard"}},
  ...
]}}"""

CONVERSATION_PROMPT = """\
You are generating multi-turn conversation training data for an AI assistant.

Given the following expert skill/prompt, generate a realistic {n}-turn conversation between a user and an AI assistant with this expertise. Each turn should build on the previous one, showing progressively deeper engagement.

Skill/Prompt:
---
{skill_content}
---

Requirements:
- The conversation should feel natural and progressive
- Each turn should add new information or go deeper
- The user should ask follow-up questions based on previous answers
- Responses should be thorough but conversational
- Extract each turn as an independent instruction-response pair (include enough context in the instruction)

Respond with a JSON object:
{{"samples": [
  {{"instruction": "user message (with enough context to stand alone)", "response": "assistant response", "difficulty": "easy|medium|hard"}},
  ...
]}}"""

EDGE_CASE_PROMPT = """\
You are generating edge-case training data for an AI assistant.

Given the following expert skill/prompt, generate {n} challenging edge cases that test the boundaries of this expertise. Include:
1. An ambiguous or vague request that needs clarification
2. A request that's partially outside the scope (needs honest boundary-setting)
3. A common misconception that needs gentle correction
4. A highly advanced/complex question that pushes expertise limits

Skill/Prompt:
---
{skill_content}
---

Requirements:
- The assistant should handle each edge case gracefully
- For ambiguous requests: ask clarifying questions while providing preliminary guidance
- For out-of-scope: clearly state limitations while being maximally helpful
- For misconceptions: correct gently with explanation
- For advanced questions: provide thorough, nuanced responses

Respond with a JSON object:
{{"samples": [
  {{"edge_type": "ambiguous|out_of_scope|misconception|advanced", "instruction": "user message", "response": "assistant response", "difficulty": "easy|medium|hard"}},
  ...
]}}"""

STRATEGY_PROMPTS = {
    "direct_task": DIRECT_TASK_PROMPT,
    "scenario_based": SCENARIO_BASED_PROMPT,
    "conversation": CONVERSATION_PROMPT,
    "edge_case": EDGE_CASE_PROMPT,
}
