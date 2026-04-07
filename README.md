# StockTrader: Competing Stock Analysis Agents

A multi-agent system that analyzes stocks using two competing investment strategies — **Momentum Trader** and **Value Contrarian** — and compares their recommendations.

## Strategies

| Strategy | Philosophy |
|----------|-----------|
| **Momentum Trader** | "The trend is your friend." Follows rising prices, increasing volume, and bullish breakout patterns. Buys winners, sells losers. |
| **Value Contrarian** | "Markets overreact. Buy fear, sell greed." Looks for oversold stocks as buying opportunities and overbought stocks as selling opportunities. |

## LLM Provider

**OpenAI GPT-4o-mini** via the OpenAI Python SDK.

## Framework

Plain Python with `concurrent.futures.ThreadPoolExecutor` for parallel strategy execution. No external agent framework required.

## Stock Selections

| Ticker | Category | Rationale |
|--------|----------|-----------|
| AAPL | Steady large-cap | Established blue-chip with consistent performance |
| NVDA | Volatile high-momentum | AI-driven growth with significant price swings |
| INTC | Recently declined | Significant downturn from competitive pressures |
| KO | Sideways/stable | Defensive consumer staple with low volatility |

## Setup and Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
# Or create a .env file in the stocktrader/ directory:
# OPENAI_API_KEY=your-key-here
```

## Running the System

```bash
cd src
python main.py                    # Run with default stocks (AAPL, NVDA, INTC, KO)
python main.py TSLA MSFT AMZN     # Run with custom stocks
```

## Pre-generated Outputs

Pre-generated JSON output files are included in the `outputs/` directory for grading without an API key. These contain the full analysis results for all four stocks.

## Project Structure

```
stocktrader/
├── README.md
├── requirements.txt
├── .env                    # Your API key (not committed)
├── src/
│   ├── main.py             # Entry point
│   ├── market_data.py      # Market data fetching (yfinance, no LLM)
│   ├── strategies.py       # Two LLM-powered strategy agents
│   ├── evaluator.py        # Evaluator component
│   └── orchestration.py    # Pipeline orchestration
├── prompts/
│   ├── strategy_a.txt      # Momentum Trader system prompt
│   ├── strategy_b.txt      # Value Contrarian system prompt
│   └── evaluator.txt       # Evaluator system prompt
├── outputs/
│   ├── AAPL.json
│   ├── NVDA.json
│   ├── INTC.json
│   ├── KO.json
│   └── summary.json
└── report/
    ├── report.pdf
    └── ai_use_appendix.pdf
```
