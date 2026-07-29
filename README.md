# Systematic Trading Backtesting Engine

A reusable Python framework for testing rule-based trading strategies using historical market data, realistic execution assumptions and detailed performance analysis.

## Overview

This project explores how systematic trading strategies can be designed, tested and compared without introducing look-ahead bias.

The framework supports several common strategies, including:

* Buy and hold
* Moving-average crossover
* Momentum
* Mean reversion

Rather than focusing only on returns, the project considers the practical factors that can significantly affect real trading performance, including transaction costs, signal delays, turnover and trade-level results.

## Start Here

Readers who are mainly interested in the ideas and results should begin with:

1. **`backtesting_engine_complete_theory_guide.ipynb`**
   Explains the theory behind backtesting, strategy signals, execution timing, transaction costs, risk metrics and common testing errors.

2. **`main.ipynb`**
   Demonstrates the complete workflow, from downloading market data and generating signals to comparing strategies and analysing results.

The remaining Python files contain the reusable implementation used by the notebook.

## Key Features

* Delays trading signals to reduce look-ahead bias
* Models transaction costs and portfolio turnover
* Tracks individual long and short trades
* Calculates Sharpe ratio, Sortino ratio, drawdown and win rate
* Compares multiple strategies using consistent assumptions
* Supports parameter optimisation and out-of-sample evaluation
* Separates strategy logic, market data and backtesting logic into reusable modules

## Purpose

The purpose of this project is not to claim that a particular strategy will remain profitable. It is to demonstrate how trading ideas can be evaluated systematically, transparently and with realistic assumptions.

This project is intended for education, research and portfolio demonstration rather than live investment decisions.
