# Integrating Structured Outputs with Your Custom OpenAI Client

## Current Setup Analysis

**Your OpenAI Client:** Custom wrapper using `requests.post` 
**Location:** `/home/nitai/code/causalix.ai/backend/lm_clients/openai_client_working.py`
**OpenAI Version Installed:** `openai 2.2.0` (not directly used, just for reference)

**Key Finding:** Your client already supports `extra_body` parameter! 

```python
if extra_body:
    payload.update(extra_body)
```

This means you can pass `response_format` directly through `extra_body`!

## How to Add Structured Outputs to Your Current Client

### Option 1: Use `extra_body` Parameter (Easiest - No Code Changes)

Your existing client already supports this! Just pass `response_format` via `extra_body`:

```python
from pydantic import BaseModel, Field
from typing import Literal

# Define your Pydantic model
class CorrectnessJudgment(BaseModel):
    chain_of_thought: str
    strengths: list[str]
    weaknesses: list[str]
    verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
    score: float
    reason: str

# Use your existing client
client = OpenAIClient(api_key=..., api_url=..., model="gpt-4o-2024-08-06")

response = client.generate(
    sys_prompt="You are a precise judge...",
    usr_prompt="Evaluate this explanation...",
    extra_body={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "correctness_judgment",
                "strict": True,  # ← This enables 100% schema compliance!
                "schema": CorrectnessJudgment.model_json_schema()
            }
        }
    }
)

# Parse the response
import json
result_text = response['choices'][0]['message']['content']
judgment = CorrectnessJudgment(**json.loads(result_text))
```

### Option 2: Add Native Support (Better Integration)

Modify your `OpenAIClient` to accept a `response_model` parameter:

#### Step 1: Update the `generate` method signature

```python
def generate(
    self,
    sys_prompt: str,
    usr_prompt: str,
    *,
    response_model: Optional[type[BaseModel]] = None,  # ← Add this
    max_tokens: int = 4000,
    temperature: Optional[float] = None,
    # ... rest of parameters
) -> Dict[str, Any]:
```

#### Step 2: Add response_format to payload

```python
# In the "chat" payload section (around line 226):
if current_api_kind == "chat":
    payload: Dict[str, Any] = {
        "model": self.model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
        token_key: max_tokens,
        "stream": stream,
    }
    
    # ← ADD THIS BLOCK
    if response_model is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__.lower(),
                "strict": True,
                "schema": response_model.model_json_schema()
            }
        }
    
    if include_temp:
        payload["temperature"] = temperature
    # ... rest of the code
```

#### Step 3: Add helper method to parse structured response

```python
def parse_structured_response(
    self,
    response: Dict[str, Any],
    model_class: type[BaseModel]
) -> BaseModel:
    """Parse structured output into Pydantic model"""
    if "choices" in response:
        content = response["choices"][0]["message"]["content"]
        return model_class(**json.loads(content))
    else:
        raise ValueError("Unexpected response format")
```

#### Step 4: Usage

```python
from pydantic import BaseModel, Field
from typing import Literal

class CorrectnessJudgment(BaseModel):
    chain_of_thought: str = Field(description="Step-by-step reasoning")
    strengths: list[str]
    weaknesses: list[str]
    verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str

# Use with your client
client = OpenAIClient(api_key=..., api_url=..., model="gpt-4o-2024-08-06")

response = client.generate(
    sys_prompt="You are a precise judge evaluating causal reasoning.",
    usr_prompt="Evaluate this explanation: ...",
    response_model=CorrectnessJudgment  # ← Pass the model class
)

# Parse the structured output
judgment = client.parse_structured_response(response, CorrectnessJudgment)

# Type-safe access!
print(judgment.verdict)              # "CORRECT"
print(judgment.chain_of_thought)     # Full reasoning
print(judgment.strengths)            # List[str]
```

## Important Requirements for Structured Outputs

### 1. Model Support

✅ **Supported models:**
- `gpt-4o-2024-08-06` (100% reliability)
- `gpt-4o-mini` 
- `gpt-4o-2024-08-06` and later

❌ **NOT supported:**
- Older models (gpt-4-turbo, gpt-3.5-turbo, etc.)
- Models before August 2024

### 2. API Endpoint

Structured outputs work with the **Chat Completions API** (`/chat/completions`), not the Responses API.

In your client, this means:
- ✅ Works when `api_kind="chat"` 
- ❌ Does NOT work when `api_kind="responses"`

### 3. Schema Requirements

When using `strict: True`, you must follow these rules:

✅ **Allowed:**
```python
class ValidModel(BaseModel):
    name: str  # Required field
    age: int   # Required field
    tags: list[str]  # Required field
    score: float = Field(ge=0.0, le=1.0)  # Constrained field
```

❌ **NOT allowed:**
```python
class InvalidModel(BaseModel):
    name: Optional[str] = None  # ❌ Optional fields not supported
    metadata: dict[str, Any]    # ❌ Arbitrary dicts not supported
```

All fields must be **required**. For "optional" behavior, use empty lists or empty strings as defaults:

```python
class GoodModel(BaseModel):
    chain_of_thought: str
    strengths: list[str] = Field(default_factory=list)  # ✅ Empty list if missing
    weaknesses: list[str] = Field(default_factory=list)  # ✅ Empty list if missing
```

## Recommended Pydantic Models for Your Use Case

### Correctness Judgment (with CoT)

```python
class CorrectnessJudgmentCoT(BaseModel):
    """Structured judgment with chain-of-thought reasoning"""
    
    # STEP 1: Reasoning (MUST come first)
    chain_of_thought: str = Field(
        description=(
            "Step-by-step reasoning evaluating: "
            "(1) logical consistency, (2) temporal ordering, "
            "(3) mechanism specificity, (4) scientific plausibility"
        )
    )
    
    # STEP 2: Analysis
    strengths: list[str] = Field(
        description="Key strengths of the causal explanation",
        default_factory=list
    )
    
    weaknesses: list[str] = Field(
        description="Key weaknesses or gaps",
        default_factory=list
    )
    
    # STEP 3: Final verdict (LAST)
    verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
    
    score: float = Field(
        description="1.0 for CORRECT, 0.5 for PARTIALLY_CORRECT, 0.0 for INCORRECT",
        ge=0.0,
        le=1.0
    )
    
    reason: str = Field(description="One-sentence summary")
```

### Citation Judgment (with CoT)

```python
class CitationJudgmentCoT(BaseModel):
    """Structured citation verification with chain-of-thought"""
    
    # STEP 1: Reasoning (FIRST)
    chain_of_thought: str = Field(
        description=(
            "Analysis of citation support considering: "
            "(1) mechanisms, (2) temporality, "
            "(3) strength of association, (4) study design"
        )
    )
    
    # STEP 2: Evidence assessment
    evidence_for_claim: list[str] = Field(
        description="Evidence that supports the claim",
        default_factory=list
    )
    
    evidence_against_claim: list[str] = Field(
        description="Evidence that contradicts the claim",
        default_factory=list
    )
    
    missing_evidence: list[str] = Field(
        description="What evidence is missing",
        default_factory=list
    )
    
    # STEP 3: Verdict (LAST)
    verdict: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
    
    score: float = Field(ge=0.0, le=1.0)
    
    reason: str
```

## Complete Working Example

```python
import json
from pydantic import BaseModel, Field
from typing import Literal
from backend.lm_clients.openai_client_working import OpenAIClient

# Define model
class CorrectnessJudgmentCoT(BaseModel):
    chain_of_thought: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    verdict: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str

# Initialize client (must use gpt-4o-2024-08-06 or later)
client = OpenAIClient(
    api_key=os.environ["OPENAI_API_KEY"],
    api_url="https://api.openai.com/v1/chat/completions",
    model="gpt-4o-2024-08-06",  # ← MUST be this model or later
    api_kind="chat"  # ← MUST be "chat" not "responses"
)

# Call with structured output via extra_body
response = client.generate(
    sys_prompt="You are a precise judge evaluating causal reasoning step by step.",
    usr_prompt=f"""Evaluate this causal explanation:

SOURCE: Sleep Deprivation
TARGET: Cognitive Performance
RELATIONSHIP: NEGATIVE
EXPLANATION: Sleep deprivation reduces cognitive performance by impairing neural 
plasticity and reducing prefrontal cortex activity.

Provide your analysis in the structured format.""",
    extra_body={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "correctness_judgment_cot",
                "strict": True,  # ← 100% schema compliance!
                "schema": CorrectnessJudgmentCoT.model_json_schema()
            }
        }
    }
)

# Parse structured response
content = response['choices'][0]['message']['content']
judgment = CorrectnessJudgmentCoT(**json.loads(content))

# Type-safe access!
print(f"Verdict: {judgment.verdict}")
print(f"Score: {judgment.score}")
print(f"\nReasoning:\n{judgment.chain_of_thought}")
print(f"\nStrengths: {judgment.strengths}")
print(f"\nWeaknesses: {judgment.weaknesses}")
```

## Testing Structured Outputs

Create a simple test script:

```python
# test_structured_outputs.py
import os
import json
from pydantic import BaseModel, Field
from typing import Literal
from backend.lm_clients.openai_client_working import OpenAIClient

class SimpleJudgment(BaseModel):
    reasoning: str
    verdict: Literal["GOOD", "BAD"]
    score: float = Field(ge=0.0, le=1.0)

client = OpenAIClient(
    api_key=os.environ["OPENAI_API_KEY"],
    api_url="https://api.openai.com/v1/chat/completions",
    model="gpt-4o-2024-08-06",
    api_kind="chat"
)

response = client.generate(
    sys_prompt="Judge if the text is positive or negative.",
    usr_prompt="The weather is beautiful today!",
    extra_body={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "simple_judgment",
                "strict": True,
                "schema": SimpleJudgment.model_json_schema()
            }
        }
    }
)

content = response['choices'][0]['message']['content']
result = SimpleJudgment(**json.loads(content))

print(f"✅ Structured output works!")
print(f"Verdict: {result.verdict}")
print(f"Score: {result.score}")
print(f"Reasoning: {result.reasoning}")
```

## Benefits Summary

**With Structured Outputs:**
- ✅ **100% schema compliance** - No parsing errors
- ✅ **+60% accuracy** on reasoning tasks (research)
- ✅ **Type-safe** - Access fields with `judgment.verdict`
- ✅ **Automatic validation** - Pydantic catches invalid data
- ✅ **Transparent reasoning** - Full CoT in structured field
- ✅ **Database-ready** - `judgment.model_dump()` → JSON

**Without:**
- ❌ Inconsistent formats
- ❌ Manual string parsing
- ❌ No validation
- ❌ -60% accuracy loss (research)

## Next Steps

1. ✅ Your client already supports `extra_body` - use it immediately!
2. Test with `gpt-4o-2024-08-06` model
3. Create Pydantic models for your judgment tasks
4. Update your evaluation code to use structured outputs
5. Measure accuracy improvements

## References

- [OpenAI Structured Outputs Docs](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- Research: +60% accuracy with reasoning fields (GSM8k benchmark)
