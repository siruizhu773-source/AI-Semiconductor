# AI-Semiconductor
Quantifying tail risk and volatility regimes for NVDA, INTC, TSM, and ^IXIC using Student-t GARCH models.

# Tail Risk & Volatility Persistence in the AI Supercycle

## Business Question
How do tail risk and volatility regimes differ across AI-semiconductor assets, and which model best captures extreme downside moves? 

## Project Overview
This repository contains the reproducible Python pipeline and the executive report for quantifying tail risk in the tech sector (2018-2023). 
* **Target Assets:** NVDA, INTC, TSM, and ^IXIC
* **Deliverable:** 2-page brief + reproducible code 

## Key Findings
* **Fat tails are real:** Non-normal returns imply Gaussian risk metrics underestimate extreme moves.
* **Volatility shocks persist:** For NVDA, volatility persistence is near unity (alpha + beta ≈ 0.974), suggesting shocks remain elevated for longer.
* **Student-t GARCH outperforms Normal GARCH:** NVDA's AIC improves from 7610.66 (Normal) to 7496.27 (Student-t), with degrees of freedom (v ≈ 6.07) supporting heavy-tails risk.

## How to Reproduce
1. Clone this repository.
2. Install dependencies: pip install -r requirements.txt
3. Run the main script: python volatility_analysis.py

## Full Report
Please navigate to the report to view the 2-page executive analytical brief.
