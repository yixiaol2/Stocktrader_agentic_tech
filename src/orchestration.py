"""
Orchestration Module
Coordinates the full pipeline: market data -> parallel strategies -> evaluator -> JSON output.
Uses concurrent.futures for parallel strategy execution.
"""

import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from market_data import fetch_all_market_data, format_market_data_for_llm, get_market_data_summary
from strategies import run_momentum_trader, run_value_contrarian, run_momentum_debate, run_contrarian_debate
from evaluator import evaluate_strategies


def analyze_stock(ticker: str, market_data: dict) -> dict:
    """
    Run the analysis pipeline for a single stock ticker (with pre-fetched market data).
    1. Format market data
    2. Run both strategy agents in parallel
    3. Evaluate agreement/disagreement
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {ticker}...")
    print(f"{'='*60}")

    market_data_text = format_market_data_for_llm(market_data)
    market_data_summary = get_market_data_summary(market_data)
    print(f"[{ticker}] Current price: ${market_data['current_price']}")

    # Run both strategies in parallel
    print(f"[{ticker}] Running strategy agents in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_momentum_trader, market_data_text)
        future_b = executor.submit(run_value_contrarian, market_data_text)
        strategy_a_result = future_a.result()
        strategy_b_result = future_b.result()

    print(f"[{ticker}] Momentum Trader: {strategy_a_result['decision']} (confidence: {strategy_a_result['confidence']})")
    print(f"[{ticker}] Value Contrarian: {strategy_b_result['decision']} (confidence: {strategy_b_result['confidence']})")

    # Evaluate
    print(f"[{ticker}] Running evaluator...")
    eval_result = evaluate_strategies(ticker, market_data_text, strategy_a_result, strategy_b_result)
    agree_status = "AGREE" if eval_result["agents_agree"] else "DISAGREE"
    print(f"[{ticker}] Strategies {agree_status}")

    result = {
        "ticker": ticker,
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "market_data_summary": market_data_summary,
        "strategy_a": strategy_a_result,
        "strategy_b": strategy_b_result,
        "evaluator": eval_result,
    }

    # Debate Mode (bonus): if strategies disagree, run a second round
    if not eval_result["agents_agree"]:
        print(f"[{ticker}] Disagreement detected — entering Debate Mode...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(run_momentum_debate, market_data_text,
                                       strategy_a_result, strategy_b_result)
            future_b = executor.submit(run_contrarian_debate, market_data_text,
                                       strategy_b_result, strategy_a_result)
            debate_a = future_a.result()
            debate_b = future_b.result()

        a_changed = debate_a["decision"] != strategy_a_result["decision"]
        b_changed = debate_b["decision"] != strategy_b_result["decision"]
        print(f"[{ticker}] Debate — Momentum Trader: {debate_a['decision']} (confidence: {debate_a['confidence']}) {'[CHANGED]' if a_changed else '[HELD FIRM]'}")
        print(f"[{ticker}] Debate — Value Contrarian: {debate_b['decision']} (confidence: {debate_b['confidence']}) {'[CHANGED]' if b_changed else '[HELD FIRM]'}")

        result["debate"] = {
            "strategy_a_revised": debate_a,
            "strategy_b_revised": debate_b,
            "a_position_changed": a_changed,
            "b_position_changed": b_changed,
            "post_debate_agree": debate_a["decision"] == debate_b["decision"],
        }

    return result


def save_result(result: dict, output_dir: str) -> None:
    """Save a single stock result to a JSON file."""
    filepath = os.path.join(output_dir, f"{result['ticker']}.json")
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {filepath}")


def generate_summary(results: list, output_dir: str) -> dict:
    """Generate and save summary.json aggregating all stock results."""
    summary = {
        "strategies": ["Momentum Trader", "Value Contrarian"],
        "stocks_analyzed": [r["ticker"] for r in results],
        "total_agreements": sum(1 for r in results if r["evaluator"]["agents_agree"]),
        "total_disagreements": sum(1 for r in results if not r["evaluator"]["agents_agree"]),
        "results": [
            {
                "ticker": r["ticker"],
                "a_decision": r["strategy_a"]["decision"],
                "b_decision": r["strategy_b"]["decision"],
                "agree": r["evaluator"]["agents_agree"],
            }
            for r in results
        ],
    }
    filepath = os.path.join(output_dir, "summary.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {filepath}")
    return summary


def run_full_analysis(tickers: list) -> list:
    """Run the full analysis pipeline for all tickers."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Batch-fetch all market data in a single API call (no LLM)
    print("Fetching market data for all tickers...")
    all_market_data = fetch_all_market_data(tickers)
    print(f"Market data fetched for: {', '.join(all_market_data.keys())}")

    # Step 2: Analyze each stock (LLM calls for strategies + evaluator)
    results = []
    for ticker in tickers:
        result = analyze_stock(ticker, all_market_data[ticker])
        save_result(result, output_dir)
        results.append(result)

    # Step 3: Generate summary
    summary = generate_summary(results, output_dir)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Strategies: Momentum Trader vs. Value Contrarian")
    print(f"Stocks analyzed: {', '.join(summary['stocks_analyzed'])}")
    print(f"Agreements: {summary['total_agreements']}")
    print(f"Disagreements: {summary['total_disagreements']}")
    print()
    for r in summary["results"]:
        status = "AGREE" if r["agree"] else "DISAGREE"
        print(f"  {r['ticker']}: Momentum={r['a_decision']}, Contrarian={r['b_decision']} -> {status}")

    return results
