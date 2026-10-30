# budget.py - Turn, tool-call, and cost limits for research runs

from dataclasses import dataclass, field
from config import MAX_SUPERVISOR_TURNS, MAX_COST_USD, PRICE_PER_1M_INPUT, PRICE_PER_1M_OUTPUT


@dataclass
class Budget:
    max_turns: int = MAX_SUPERVISOR_TURNS
    max_cost_usd: float = MAX_COST_USD

    turns_used: int = 0
    tool_calls_used: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def record_turn(self):
        self.turns_used += 1

    def record_tool_calls(self, n: int):
        self.tool_calls_used += n

    def record_tokens(self, tokens_in: int, tokens_out: int):
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    @property
    def cost_usd(self) -> float:
        return (
            self.tokens_in * PRICE_PER_1M_INPUT / 1_000_000
            + self.tokens_out * PRICE_PER_1M_OUTPUT / 1_000_000
        )

    def exhausted(self) -> str | None:
        """Return the reason the budget is exhausted, or None if still within limits."""
        if self.turns_used >= self.max_turns:
            return "turn_limit"
        if self.cost_usd >= self.max_cost_usd:
            return "cost_limit"
        return None

    def remaining_summary(self) -> str:
        return (
            f"Turns: {self.turns_used}/{self.max_turns} | "
            f"Cost: ${self.cost_usd:.4f}/${self.max_cost_usd:.2f} | "
            f"Tool calls: {self.tool_calls_used} | "
            f"Tokens: {self.tokens_in}in/{self.tokens_out}out"
        )

    def to_dict(self) -> dict:
        return {
            "max_turns": self.max_turns,
            "max_cost_usd": self.max_cost_usd,
            "turns_used": self.turns_used,
            "tool_calls_used": self.tool_calls_used,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Budget":
        return cls(
            max_turns=data.get("max_turns", MAX_SUPERVISOR_TURNS),
            max_cost_usd=data.get("max_cost_usd", MAX_COST_USD),
            turns_used=data.get("turns_used", 0),
            tool_calls_used=data.get("tool_calls_used", 0),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
        )
