"""
Strategy Agents
Two LLM-powered strategy agents with distinct behavioral philosophies.
Each agent receives the same market data and produces independent recommendations.
"""

import json
import os
from openai import OpenAI


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    with open(os.path.join(prompts_dir, filename), "r") as f:
        return f.read()


def _call_llm(system_prompt: str, user_message: str) -> dict:
    """Make an LLM call and parse the JSON response."""
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
    return json.loads(content)


def run_momentum_trader(market_data_text: str) -> dict:
    """
    Run the Momentum Trader strategy agent.
    Returns: {"name": str, "decision": str, "confidence": int, "justification": str}
    """
    system_prompt = _load_prompt("strategy_a.txt")
    result = _call_llm(system_prompt, f"Analyze the following stock data:\n\n{market_data_text}")
    return {
        "name": "Momentum Trader",
        "decision": result["decision"].upper(),
        "confidence": int(result["confidence"]),
        "justification": result["justification"],
    }


def run_value_contrarian(market_data_text: str) -> dict:
    """
    Run the Value Contrarian strategy agent.
    Returns: {"name": str, "decision": str, "confidence": int, "justification": str}
    """
    system_prompt = _load_prompt("strategy_b.txt")
    result = _call_llm(system_prompt, f"Analyze the following stock data:\n\n{market_data_text}")
    return {
        "name": "Value Contrarian",
        "decision": result["decision"].upper(),
        "confidence": int(result["confidence"]),
        "justification": result["justification"],
    }


# --- Debate Mode (Bonus Extension) ---

DEBATE_PROMPT = """You previously analyzed a stock and gave your recommendation. Now you have seen
the opposing strategy agent's reasoning. Consider their arguments carefully.

You may revise your decision, adjust your confidence, or hold firm — but you MUST explain why.
Reference specific points from the opponent's argument that you agree with, disagree with, or find irrelevant to your philosophy.

Your original analysis:
  Decision: {my_decision}
  Confidence: {my_confidence}/10
  Justification: {my_justification}

The opposing agent ({opponent_name}) argued:
  Decision: {opp_decision}
  Confidence: {opp_confidence}/10
  Justification: {opp_justification}

Based on the market data and the opponent's reasoning, provide your revised (or reaffirmed) position.

Respond in this exact JSON format and nothing else:
{{
  "decision": "BUY" or "HOLD" or "SELL",
  "confidence": <integer 1-10>,
  "justification": "<3-5 sentences explaining whether and why you changed or held firm, referencing the opponent's argument>"
}}"""


def run_debate_round(strategy_prompt_file: str, strategy_name: str,
                     my_result: dict, opponent_result: dict,
                     market_data_text: str) -> dict:
    """
    Run a debate round where an agent responds to the opponent's reasoning.
    Returns the revised (or reaffirmed) position.
    """
    system_prompt = _load_prompt(strategy_prompt_file)

    debate_message = DEBATE_PROMPT.format(
        my_decision=my_result["decision"],
        my_confidence=my_result["confidence"],
        my_justification=my_result["justification"],
        opponent_name=opponent_result["name"],
        opp_decision=opponent_result["decision"],
        opp_confidence=opponent_result["confidence"],
        opp_justification=opponent_result["justification"],
    )

    user_message = f"Market data:\n\n{market_data_text}\n\n{debate_message}"
    result = _call_llm(system_prompt, user_message)

    return {
        "name": strategy_name,
        "decision": result["decision"].upper(),
        "confidence": int(result["confidence"]),
        "justification": result["justification"],
    }


def run_momentum_debate(market_data_text: str, my_result: dict, opponent_result: dict) -> dict:
    """Momentum Trader responds to Value Contrarian's reasoning."""
    return run_debate_round("strategy_a.txt", "Momentum Trader",
                            my_result, opponent_result, market_data_text)


def run_contrarian_debate(market_data_text: str, my_result: dict, opponent_result: dict) -> dict:
    """Value Contrarian responds to Momentum Trader's reasoning."""
    return run_debate_round("strategy_b.txt", "Value Contrarian",
                            my_result, opponent_result, market_data_text)
