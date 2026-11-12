# platform/agent_registry.py - Agent registry: register, discover, health-check
#
# Tracks available agents, their capabilities, tools, and health status.
# Supports dynamic registration/deregistration at runtime.

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class AgentStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    DEREGISTERED = "deregistered"


@dataclass
class RegisteredAgent:
    agent_id: str
    name: str
    description: str
    tools: list[str]
    capabilities: list[str]
    status: AgentStatus = AgentStatus.HEALTHY
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class AgentRegistry:
    """In-memory agent registry with health checking."""

    HEARTBEAT_TIMEOUT_S = 60  # agent is unhealthy if no heartbeat for 60s

    def __init__(self):
        self._agents: dict[str, RegisteredAgent] = {}

    def register(
        self,
        agent_id: str,
        name: str,
        description: str,
        tools: list[str],
        capabilities: list[str],
        metadata: dict | None = None,
    ) -> RegisteredAgent:
        agent = RegisteredAgent(
            agent_id=agent_id,
            name=name,
            description=description,
            tools=tools,
            capabilities=capabilities,
            metadata=metadata or {},
        )
        self._agents[agent_id] = agent
        return agent

    def deregister(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        agent.status = AgentStatus.DEREGISTERED
        return True

    def heartbeat(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if not agent or agent.status == AgentStatus.DEREGISTERED:
            return False
        agent.last_heartbeat = time.time()
        agent.status = AgentStatus.HEALTHY
        return True

    def get(self, agent_id: str) -> Optional[RegisteredAgent]:
        return self._agents.get(agent_id)

    def list_all(self, include_deregistered: bool = False) -> list[RegisteredAgent]:
        agents = list(self._agents.values())
        if not include_deregistered:
            agents = [a for a in agents if a.status != AgentStatus.DEREGISTERED]
        return agents

    def discover_by_capability(self, capability: str) -> list[RegisteredAgent]:
        return [
            a for a in self._agents.values()
            if capability in a.capabilities and a.status == AgentStatus.HEALTHY
        ]

    def discover_by_tool(self, tool_name: str) -> list[RegisteredAgent]:
        return [
            a for a in self._agents.values()
            if tool_name in a.tools and a.status == AgentStatus.HEALTHY
        ]

    def check_health(self) -> dict[str, str]:
        """Check all agents and mark unhealthy if heartbeat timeout exceeded."""
        now = time.time()
        results = {}
        for agent in self._agents.values():
            if agent.status == AgentStatus.DEREGISTERED:
                continue
            if now - agent.last_heartbeat > self.HEARTBEAT_TIMEOUT_S:
                agent.status = AgentStatus.UNHEALTHY
            results[agent.agent_id] = agent.status.value
        return results

    def to_dict(self) -> dict:
        return {
            "agents": [a.to_dict() for a in self._agents.values()],
            "total": len(self._agents),
            "healthy": sum(1 for a in self._agents.values() if a.status == AgentStatus.HEALTHY),
        }


# Singleton registry with pre-registered research agents
_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    return _registry


def register_default_agents() -> None:
    """Register the built-in research agents."""
    _registry.register(
        agent_id="paper_agent",
        name="Paper Agent",
        description="Searches ArXiv and Semantic Scholar for academic papers",
        tools=["search_arxiv", "search_semantic_scholar", "get_references", "get_citations"],
        capabilities=["academic_research", "citation_graph", "paper_search"],
    )
    _registry.register(
        agent_id="web_agent",
        name="Web Agent",
        description="Searches the web and GitHub for implementations and reports",
        tools=["search_web", "search_github", "read_url"],
        capabilities=["web_research", "code_search", "url_reading"],
    )
    _registry.register(
        agent_id="fact_checker",
        name="Fact Checker",
        description="Verifies claims via deterministic quote matching and LLM entailment",
        tools=[],
        capabilities=["fact_checking", "evidence_verification"],
    )
    _registry.register(
        agent_id="writer",
        name="Writer",
        description="Produces structured research reports with validated citations",
        tools=[],
        capabilities=["report_generation", "citation_validation"],
    )
    _registry.register(
        agent_id="supervisor",
        name="Supervisor",
        description="LLM-based routing coordinator that orchestrates worker agents",
        tools=[],
        capabilities=["orchestration", "routing", "budget_management"],
    )
