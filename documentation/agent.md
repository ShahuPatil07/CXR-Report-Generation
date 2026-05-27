# Agentic Report Generation — Build Guide

## What this adds to the project

Everything built in Weeks 1–4 is a **pipeline**: image goes in, report comes out, no reasoning in between. The fine-tuned LLaVA model generates a report in one shot — it doesn't check its own work, doesn't verify that what it wrote is consistent with what it saw, and doesn't think about which findings to look for before writing.

This component wraps that pipeline inside an **agent** that reasons before it writes:

1. Look at the image and classify what findings are likely present
2. For each high-confidence finding, gather visual evidence (where in the image?)
3. Draft a report using the fine-tuned model, seeded with the classifier findings as context
4. Review the draft and revise if the draft contradicts the classifier evidence

The key difference: the model is no longer forced to generate a report in one shot. It can gather information, reason about it, and then write. This directly targets the hallucination problem.

---

## Architecture

```
User input: CXR image path
       │
       ▼
┌──────────────────────────────────────┐
│   Claude (claude-opus-4-7)           │
│   Orchestrator — decides what to do  │
│   Has access to 3 tools              │
└────────────┬─────────────────────────┘
             │ calls tools in reasoning order
    ┌────────┼────────────────────────┐
    ▼        ▼                        ▼
classify_  get_visual_           draft_report
image()    evidence()            ()
    │        │                        │
    │  CheXpert classifier    Fine-tuned LLaVA
    │  (Week 4)               (Week 2 best_checkpoint)
    │        │                        │
    │  Attention rollout              │
    │  (Week 4)                       │
    └────────┴────────────────────────┘
             │ tool results returned as JSON text
             ▼
┌──────────────────────────────────────┐
│   Claude reasons over tool outputs   │
│   Writes final structured report     │
└──────────────────────────────────────┘
       │
       ▼
Final report (FINDINGS: ... IMPRESSION: ...)
```

**Why Claude as orchestrator and not LLaVA?** LLaVA is a generative model — it's good at image-conditioned text generation. It is not good at tool use, multi-step reasoning, or JSON-structured outputs. Claude is built for exactly that. The division of labour: LLaVA does the medical vision work, Claude does the reasoning and coordination.

---

## How Claude Tool Use Works

Before writing any code, understand the tool use loop. It is a back-and-forth between your code and Claude, not a single API call.

**The loop:**

```
Round 1:
  You → Claude: "Here is a CXR image path. Generate a report."
                 + tool definitions (classify_image, get_visual_evidence, draft_report)
  Claude → You: [tool_use block: {"name": "classify_image", "input": {"image_path": "..."}]
                 stop_reason = "tool_use"

Round 2:
  You execute classify_image() locally, get result
  You → Claude: [tool_result block: {"findings": [...]}]
  Claude → You: [tool_use block: {"name": "get_visual_evidence", "input": {...}}]
                 stop_reason = "tool_use"

Round 3:
  You execute get_visual_evidence() locally, get result
  You → Claude: [tool_result block: {...}]
  Claude → You: [tool_use block: {"name": "draft_report", "input": {...}}]
                 stop_reason = "tool_use"

Round 4:
  You execute draft_report() locally, get result
  You → Claude: [tool_result block: {"draft": "FINDINGS: ..."}]
  Claude → You: "FINDINGS: Bilateral pleural effusion..."
                 stop_reason = "end_turn"
```

The agent loop runs until `stop_reason == "end_turn"`. In each round, you check if the response contains `tool_use` blocks, execute those tools, and feed the results back. Claude handles all the reasoning — you just execute what it asks.

**SDK pattern:** Use the Anthropic Python SDK. The relevant classes are `anthropic.Anthropic()`, `client.messages.create()`. Tool definitions go in the `tools` parameter. Tool results are sent as messages with `role="user"` containing a `tool_result` content block with the matching `tool_use_id`.

Read the tool use docs before writing code: https://docs.anthropic.com/en/docs/tool-use

---

## File to Create: `agent.py`

Create this at the project root. It should contain:

### 1. The three tool functions

These are regular Python functions. Each takes a small number of arguments, does the heavy lifting, and returns a plain Python dict (you'll serialize to JSON when sending back to Claude).

---

**`classify_image(image_path: str) -> dict`**

Loads your Week 4 CheXpert classifier and the frozen ViT. Runs inference on the image. Returns a dict with the 14 label names and their confidence scores (after sigmoid, so values are 0–1). Only return findings above a threshold — 0.3 is a reasonable starting point.

Return format suggestion:
```
{
  "findings": [
    {"label": "Pleural Effusion", "confidence": 0.87},
    {"label": "Cardiomegaly", "confidence": 0.61}
  ],
  "normal": false
}
```

If no finding exceeds the threshold, set `"normal": true` and return an empty findings list.

**Implementation note:** Loading the model is slow. Cache the classifier and ViT as module-level globals so they are loaded once when `agent.py` is imported, not on every tool call.

---

**`get_visual_evidence(image_path: str, finding: str) -> dict`**

Runs your Week 4 attention rollout for the given image. Returns a text description of which anatomical region shows the highest activation — not the raw heatmap (Claude can't receive numpy arrays as tool results, only text/JSON).

To convert the heatmap to text:
- Divide the 24×24 patch grid into anatomical quadrants: upper-left, upper-right, lower-left, lower-right, center
- Find which quadrant contains the highest mean attention value
- Map to anatomical name: lower-left/right → costophrenic angles, center → cardiac silhouette and mediastinum, upper → lung apices
- Also report the peak activation value as a rough confidence

Return format suggestion:
```
{
  "finding": "Pleural Effusion",
  "primary_region": "left lower lung / costophrenic angle",
  "peak_activation": 0.73,
  "interpretation": "Model attention concentrated in lower-left region, consistent with left pleural effusion location"
}
```

**Implementation note:** Attention rollout requires a forward pass through the full model. You need the LLaVA model loaded. Cache it.

---

**`draft_report(image_path: str, confirmed_findings: list[str]) -> dict`**

Loads your fine-tuned LLaVA model (Week 2 best checkpoint) and generates a report. The key difference from plain inference: include `confirmed_findings` in the prompt to ground the generation.

Modify the instruction like this:
```
"Clinical context: The following findings have been detected with high confidence: 
{', '.join(confirmed_findings)}. 
Generate a detailed radiology report consistent with these findings."
```

This is called **context injection** — you're seeding the LLM with structured information before it generates free text. It significantly reduces hallucination because the model's generation is anchored to the classifier's output.

Return format:
```
{
  "draft": "FINDINGS: ...\nIMPRESSION: ..."
}
```

---

### 2. The tool definitions (JSON Schema)

For each function, write a JSON schema that describes it to Claude. The schema has three parts: `name`, `description`, and `input_schema`. The `input_schema` follows JSON Schema format — specify each parameter's type, description, and whether it's required.

Write the description fields carefully. Claude reads them to decide when and how to call the tool. A vague description leads to poor tool selection. Be specific: "Call this tool first, before any others, to identify which pathologies are present."

---

### 3. The system prompt

Write a system prompt that tells Claude:
- It is a radiology report generation assistant
- It should always call `classify_image` first to understand what findings to look for
- For any finding with confidence > 0.5, it should call `get_visual_evidence` to confirm location
- It should call `draft_report` with the confirmed findings as context
- After receiving the draft, it should review it against the classifier output and correct any inconsistencies
- The final report must follow the FINDINGS / IMPRESSION structure

The system prompt is where your clinical knowledge goes. Think: what would a radiologist's mental workflow look like? Encode that as instructions to Claude.

---

### 4. The agent loop function

Write `run_agent(image_path: str) -> str` which:

1. Initializes the Anthropic client
2. Builds the initial messages list with the image path in the user message
3. Calls `client.messages.create()` with `model`, `system`, `tools`, `messages`, `max_tokens`
4. Enters a while loop: while `response.stop_reason == "tool_use"`:
   - Extract all `tool_use` content blocks from the response
   - For each tool_use block, call the matching Python function with `block.input`
   - Append the assistant response and the tool results to `messages`
   - Call `client.messages.create()` again with the updated messages
5. When `stop_reason == "end_turn"`, extract and return the final text response

**Handling multiple tool calls in one round:** Claude can request multiple tools in a single response (e.g., call `get_visual_evidence` for both pleural effusion and cardiomegaly at once). Your loop must handle a list of tool_use blocks, execute all of them, and send all results back in one `user` message containing multiple `tool_result` blocks.

---

## Evaluating the Agent

Run the agent on the same 200 test samples used in `baseline.ipynb`. Compare:

| Model | BLEU-4 | ROUGE-L | BERTScore F1 |
|---|---|---|---|
| Base (zero-shot) | ? | ? | ? |
| SFT (Week 2) | ? | ? | ? |
| DPO (Week 3) | ? | ? | ? |
| Agent (this) | ? | ? | ? |

The agent should beat DPO on BERTScore F1 and RadGraph F1 (semantic/clinical metrics) because its outputs are grounded in classifier evidence. It may not beat DPO on BLEU/ROUGE (surface metrics) because the report wording is influenced by Claude rather than purely the fine-tuned LLaVA.

**Additional metric — hallucination rate:**

This is the agent's killer metric. For each generated report, use the CheXpert labeler to extract findings from the text, then compare against the classifier's predictions. Count cases where the report claims a finding the classifier scored < 0.2. This is your hallucination rate.

Expected result: agent should have a significantly lower hallucination rate than plain LLaVA generation, because the classifier acts as a gate on what findings are allowed to appear in the draft.

---

## Integration with the Rest of the Project

Add the agent to your `final_eval.ipynb` as the fourth model. The comparison table becomes:

```
Base VLM → SFT → DPO → Agent
```

Each step represents a different level of the system:
- Base → SFT: domain adaptation (language)
- SFT → DPO: preference alignment (quality)
- DPO → Agent: structured reasoning (grounding)

In the write-up, this becomes Section 4.4: "Agentic Grounding." The narrative is that each step addresses a different failure mode of the previous one.

---

## Practical Notes

**Cost:** Each agent run calls Claude once per round, typically 3–4 rounds. At `claude-opus-4-7` pricing (~$5/1M input tokens), running 200 test samples through the agent will cost roughly $3–8 depending on prompt length. Budget for this.

**Latency:** Each agent run takes 3–4 API calls + 3 model inference calls (classifier, attention rollout, LLaVA). Expect 15–30 seconds per image. For 200 samples that's ~1 hour. Run it overnight.

**Caching model weights:** The most important performance optimization. Load all three models (classifier, LLaVA) once at startup and reuse them. Don't reload from disk on every tool call.

**Error handling:** Wrap each tool call in a try/except. If the classifier fails on one image, return `{"error": "classifier failed", "findings": []}`. Claude is smart enough to proceed with only the draft_report tool in that case. Without error handling, one bad image crashes the entire 200-sample evaluation run.

---

## Self-Check Questions

- What is the difference between a tool definition (the JSON schema you give Claude) and a tool implementation (the Python function you write)? Why are they separate?
- Why does `draft_report` take `confirmed_findings` as a parameter? What is the theoretical reason this reduces hallucination compared to generating without this context?
- In the agent loop, when Claude returns `stop_reason="tool_use"`, you must append the assistant message to `messages` before sending tool results. What happens if you skip this step?
- The agent calls `classify_image` once, then potentially multiple calls to `get_visual_evidence` (one per finding), then `draft_report` once. Claude decides this order. What would you change in the system prompt if you wanted Claude to always call all three tools, never skip any?
- Why is `hallucination rate` a better metric for evaluating the agent than ROUGE-L?
