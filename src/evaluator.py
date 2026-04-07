"""
Evaluator Component
Compares the two strategy agents' recommendations and produces
a consensus summary or disagreement analysis.
"""

import json
import os
from openai import OpenAI


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    with open(os.path.join(prompts_dir, filename), "r") as f:
        return f.read()


def evaluate_strategies(
    ticker: str,
    market_data_text: str,
    strategy_a: dict,
    strategy_b: dict,
) -> dict:
    """
    Compare both strategy outputs and produce an evaluation.
    If decisions match: consensus summary.
    If decisions differ: disagreement analysis.

    Returns: {"agents_agree": bool, "analysis": str}
    """
    agents_agree = strategy_a["decision"] == strategy_b["decision"]

    system_prompt = _load_prompt("evaluator.txt")

    user_message = f"""Stock: {ticker}

Market Data:
{market_data_text}

Strategy A — {strategy_a['name']}:
  Decision: {strategy_a['decision']}
  Confidence: {strategy_a['confidence']}/10
  Justification: {strategy_a['justification']}

Strategy B — {strategy_b['name']}:
  Decision: {strategy_b['decision']}
  Confidence: {strategy_b['confidence']}/10
  Justification: {strategy_b['justification']}

The two strategies {"AGREE" if agents_agree else "DISAGREE"} on their decision.
Please provide your {"consensus summary" if agents_agree else "disagreement analysis"}."""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    result = json.loads(content)

    return {
        "agents_agree": agents_agree,
        "analysis": result["analysis"],
    }
