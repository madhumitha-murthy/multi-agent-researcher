# tests/test_platform.py - Tests for auth, RBAC, agent registry, and gateway

import time

import pytest
from fastapi.testclient import TestClient

from research_platform.agent_registry import AgentRegistry, AgentStatus
from research_platform.auth import (
    Role,
    authenticate_user,
    create_access_token,
    decode_token,
    has_permission,
    register_agent_identity,
    verify_password,
)
from research_platform.logging_config import correlation_id, new_correlation_id


# ==================== AUTH TESTS ====================

class TestAuth:
    def test_password_hashing(self):
        from research_platform.auth import pwd_context
        hashed = pwd_context.hash("testpass")
        assert verify_password("testpass", hashed)
        assert not verify_password("wrongpass", hashed)

    def test_authenticate_user_success(self):
        user = authenticate_user("admin", "admin")
        assert user is not None
        assert user.role == Role.ADMIN

    def test_authenticate_user_wrong_password(self):
        assert authenticate_user("admin", "wrong") is None

    def test_authenticate_user_unknown(self):
        assert authenticate_user("nobody", "pass") is None

    def test_create_and_decode_token(self):
        token = create_access_token(sub="testuser", role=Role.RESEARCHER)
        data = decode_token(token)
        assert data.sub == "testuser"
        assert data.role == Role.RESEARCHER
        assert "research" in data.scopes
        assert "view" in data.scopes

    def test_admin_has_all_permissions(self):
        token = create_access_token(sub="admin", role=Role.ADMIN)
        data = decode_token(token)
        assert has_permission(data, "research")
        assert has_permission(data, "manage_agents")
        assert has_permission(data, "manage_users")

    def test_viewer_cannot_research(self):
        token = create_access_token(sub="viewer", role=Role.VIEWER)
        data = decode_token(token)
        assert has_permission(data, "view")
        assert not has_permission(data, "research")
        assert not has_permission(data, "manage_agents")

    def test_agent_identity_token(self):
        token = register_agent_identity("test_agent", "Test Agent", Role.RESEARCHER)
        data = decode_token(token)
        assert data.sub == "agent:test_agent"
        assert data.is_agent is True
        assert data.role == Role.RESEARCHER


# ==================== AGENT REGISTRY TESTS ====================

class TestAgentRegistry:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_register_agent(self):
        agent = self.registry.register(
            agent_id="test1", name="Test", description="A test agent",
            tools=["tool_a"], capabilities=["cap_a"],
        )
        assert agent.agent_id == "test1"
        assert agent.status == AgentStatus.HEALTHY

    def test_list_agents(self):
        self.registry.register("a1", "A1", "desc", ["t1"], ["c1"])
        self.registry.register("a2", "A2", "desc", ["t2"], ["c2"])
        assert len(self.registry.list_all()) == 2

    def test_deregister_agent(self):
        self.registry.register("a1", "A1", "desc", [], [])
        assert self.registry.deregister("a1") is True
        assert len(self.registry.list_all()) == 0  # excluded by default
        assert len(self.registry.list_all(include_deregistered=True)) == 1

    def test_discover_by_capability(self):
        self.registry.register("a1", "A1", "desc", [], ["search"])
        self.registry.register("a2", "A2", "desc", [], ["write"])
        results = self.registry.discover_by_capability("search")
        assert len(results) == 1
        assert results[0].agent_id == "a1"

    def test_discover_by_tool(self):
        self.registry.register("a1", "A1", "desc", ["search_arxiv"], [])
        results = self.registry.discover_by_tool("search_arxiv")
        assert len(results) == 1

    def test_heartbeat(self):
        self.registry.register("a1", "A1", "desc", [], [])
        assert self.registry.heartbeat("a1") is True
        assert self.registry.heartbeat("nonexistent") is False

    def test_health_check_timeout(self):
        self.registry.register("a1", "A1", "desc", [], [])
        agent = self.registry.get("a1")
        # Simulate old heartbeat
        agent.last_heartbeat = time.time() - 120
        health = self.registry.check_health()
        assert health["a1"] == "unhealthy"

    def test_deregistered_agent_no_heartbeat(self):
        self.registry.register("a1", "A1", "desc", [], [])
        self.registry.deregister("a1")
        assert self.registry.heartbeat("a1") is False


# ==================== LOGGING TESTS ====================

class TestLogging:
    def test_correlation_id(self):
        cid = new_correlation_id()
        assert len(cid) == 8
        assert correlation_id.get() == cid

    def test_new_id_replaces_old(self):
        cid1 = new_correlation_id()
        cid2 = new_correlation_id()
        assert cid1 != cid2
        assert correlation_id.get() == cid2


# ==================== GATEWAY INTEGRATION TESTS ====================

class TestGateway:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        from research_platform.agent_registry import register_default_agents
        register_default_agents()
        from research_platform.gateway import app
        self.client = TestClient(app)

    def _get_token(self, username="admin", password="admin") -> str:
        resp = self.client.post("/auth/token", data={
            "username": username, "password": password,
        })
        assert resp.status_code == 200
        return resp.json()["access_token"]

    def test_health_no_auth(self):
        resp = self.client.get("/api/research/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_login_success(self):
        resp = self.client.post("/auth/token", data={
            "username": "admin", "password": "admin",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["role"] == "admin"

    def test_login_failure(self):
        resp = self.client.post("/auth/token", data={
            "username": "admin", "password": "wrong",
        })
        assert resp.status_code == 401

    def test_me_endpoint(self):
        token = self._get_token()
        resp = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["username"] == "admin"
        assert resp.json()["role"] == "admin"

    def test_unauthorized_without_token(self):
        resp = self.client.get("/api/agents")
        assert resp.status_code == 401

    def test_list_agents_as_admin(self):
        token = self._get_token()
        resp = self.client.get("/api/agents", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert len(agents) >= 4  # paper_agent, web_agent, fact_checker, writer

    def test_list_agents_as_viewer(self):
        token = self._get_token("viewer", "viewer")
        resp = self.client.get("/api/agents", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200

    def test_register_agent_as_admin(self):
        token = self._get_token()
        resp = self.client.post("/api/agents", json={
            "agent_id": "custom_agent",
            "name": "Custom Agent",
            "description": "A custom agent",
            "tools": ["custom_tool"],
            "capabilities": ["custom_cap"],
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 201

    def test_register_agent_forbidden_for_viewer(self):
        token = self._get_token("viewer", "viewer")
        resp = self.client.post("/api/agents", json={
            "agent_id": "x", "name": "X", "description": "x",
            "tools": [], "capabilities": [],
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_register_agent_forbidden_for_researcher(self):
        token = self._get_token("researcher", "researcher")
        resp = self.client.post("/api/agents", json={
            "agent_id": "x", "name": "X", "description": "x",
            "tools": [], "capabilities": [],
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_discover_agents_by_capability(self):
        token = self._get_token()
        resp = self.client.get(
            "/api/agents/discover/academic_research",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        agents = resp.json()["agents"]
        assert any(a["agent_id"] == "paper_agent" for a in agents)

    def test_agent_heartbeat(self):
        token = self._get_token()
        resp = self.client.post(
            "/api/agents/paper_agent/heartbeat",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_create_agent_identity(self):
        token = self._get_token()
        resp = self.client.post("/api/agent-identities", json={
            "agent_id": "new_agent",
            "agent_name": "New Agent",
            "role": "researcher",
        }, headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 201
        assert "token" in resp.json()

    def test_correlation_id_in_response(self):
        resp = self.client.get("/api/research/health")
        assert "X-Correlation-ID" in resp.headers
        assert len(resp.headers["X-Correlation-ID"]) == 8

    def test_get_single_agent(self):
        token = self._get_token()
        resp = self.client.get(
            "/api/agents/paper_agent",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Paper Agent"

    def test_get_nonexistent_agent(self):
        token = self._get_token()
        resp = self.client.get(
            "/api/agents/nonexistent",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
