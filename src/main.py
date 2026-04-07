"""
StockTrader: Building Competing Stock Analysis Agents
Main entry point - runs the full analysis pipeline on selected stocks.

Usage:
    python main.py                    # Run with default stocks
    python main.py AAPL NVDA INTC KO  # Run with custom stocks
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from orchestration import run_full_analysis


# Default stock selections:
# AAPL - Steady, established large-cap
# NVDA - Volatile, high-momentum (AI boom)
# INTC - Recently declined significantly
# KO   - Trading sideways, stable
DEFAULT_TICKERS = ["AAPL", "NVDA", "INTC", "KO"]


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS

    print("StockTrader: Momentum Trader vs. Value Contrarian")
    print(f"Analyzing: {', '.join(tickers)}")
    print(f"LLM: OpenAI GPT-4o-mini")
    print()

    results = run_full_analysis(tickers)

    print(f"\nAnalysis complete. {len(results)} stocks processed.")
    print("Output files saved to the outputs/ directory.")


if __name__ == "__main__":
    main()
