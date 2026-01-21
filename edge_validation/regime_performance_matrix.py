from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class RegimeCell:
    profit: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0


class RegimePerformanceMatrix:
    """
    Build a strategy x regime performance matrix from trade history.
    Trades must contain: 'strategy', 'regime', 'pnl'.
    """

    def __init__(self, trades: List[Dict[str, Any]]):
        self.trades = trades or []

    def build(self) -> Dict[str, Dict[str, RegimeCell]]:
        matrix: Dict[str, Dict[str, RegimeCell]] = {}
        for t in self.trades:
            strat = t.get('strategy', 'unknown')
            regime = (t.get('regime') or 'ANY')
            pnl = float(t.get('pnl', 0.0))
            matrix.setdefault(strat, {})
            cell = matrix[strat].setdefault(regime, RegimeCell())
            cell.profit += pnl
            cell.trades += 1
        # Post-process
        for strat, regs in matrix.items():
            for regime, cell in regs.items():
                if cell.trades > 0:
                    # For simplicity, win_rate as positive pnl proportion
                    # and expectancy as avg pnl
                    cell.expectancy = cell.profit / cell.trades
                    # recompute win rate
                    wins = sum(1 for t in self.trades if t.get('strategy') == strat and (t.get('regime') or 'ANY') == regime and float(t.get('pnl', 0.0)) > 0)
                    cell.win_rate = wins / cell.trades if cell.trades > 0 else 0.0
        return matrix
