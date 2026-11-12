# platform/gateway.py - FastAPI API Gateway
#
# Features:
# - OAuth2 bearer token authentication
# - RBAC-protected endpoints
# - Rate limiting middleware
# - Agent registry REST API
# - Research orchestration endpoint
# - Structured logging with correlation IDs

import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from pydantic import BaseModel

from research_platform.agent_registry import (
    AgentRegistry,
    get_registry,
    register_default_agents,
)
from research_platform.auth import (
    Role,
    TokenData,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    has_permission,
    list_agent_identities,
    register_agent_identity,
)
from research_platform.logging_config import get_logger, new_correlation_id, setup_logging

logger = get_logger("gateway")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# --- Rate limiting ---

class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, requests_per_minute: int = 30):
        self._rpm = requests_per_minute
        self._windows: dict[str, list[float]] = {}

    def check(self, key: str) -> bool:
        now = time.time()
        window = self._windows.setdefault(key, [])
        # Remove entries older than 60s
        self._windows[key] = [t for t in window if now - t < 60]
        if len(self._windows[key]) >= self._rpm:
            return False
        self._windows[key].append(now)
        return True


_rate_limiter = RateLimiter()


# --- Dependency: get current user from JWT ---

async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    try:
        token_data = decode_token(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


def require_permission(permission: str):
    """Dependency factory: require a specific permission."""
    async def checker(user: TokenData = Depends(get_current_user)):
        if not has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: requires '{permission}'",
            )
        return user
    return checker


# --- Request/Response models ---

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: str
    expires_in: int


class ResearchRequest(BaseModel):
    question: str


class AgentRegisterRequest(BaseModel):
    agent_id: str
    name: str
    description: str
    tools: list[str]
    capabilities: list[str]


class AgentIdentityRequest(BaseModel):
    agent_id: str
    agent_name: str
    role: str = "researcher"


# --- App lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    register_default_agents()
    # Register agent identities for built-in agents
    for agent in get_registry().list_all():
        register_agent_identity(agent.agent_id, agent.name)
    logger.info("Gateway started, agents registered")
    yield
    logger.info("Gateway shutting down")


# --- FastAPI app ---

app = FastAPI(
    title="Multi-Agent Research Platform",
    description="API Gateway with OAuth2, RBAC, and agent orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Middleware: correlation ID + rate limiting ---

@app.middleware("http")
async def correlation_and_rate_limit(request: Request, call_next):
    cid = new_correlation_id()
    request.state.correlation_id = cid

    # Rate limit by IP (skip auth endpoints)
    if not request.url.path.startswith("/auth"):
        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.check(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

    t0 = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - t0) * 1000)

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code}",
        extra={
            "duration_ms": duration_ms,
            "status_code": response.status_code,
        },
    )
    response.headers["X-Correlation-ID"] = cid
    return response


# ==================== AUTH ENDPOINTS ====================

@app.post("/auth/token", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow: exchange username/password for JWT."""
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from research_platform.auth import ACCESS_TOKEN_EXPIRE_MINUTES
    access_token = create_access_token(sub=user.username, role=user.role)
    refresh_token = create_refresh_token(sub=user.username)

    logger.info(f"User authenticated: {user.username}", extra={"user": user.username})
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        role=user.role.value,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.get("/auth/me")
async def get_me(user: TokenData = Depends(get_current_user)):
    """Get current user info from JWT."""
    return {
        "username": user.sub,
        "role": user.role.value,
        "scopes": user.scopes,
        "is_agent": user.is_agent,
    }


# ==================== RESEARCH ENDPOINTS ====================

@app.post("/api/research")
async def start_research(
    req: ResearchRequest,
    user: TokenData = Depends(require_permission("research")),
):
    """Start a research task (requires 'research' permission)."""
    from dotenv import load_dotenv
    load_dotenv()

    from graph import build_graph, run_research
    from llm import GeminiModel

    logger.info(
        f"Research started: {req.question[:80]}",
        extra={"user": user.sub, "action": "research_start"},
    )

    model = GeminiModel()
    graph = build_graph(model)
    result = run_research(graph, req.question)

    logger.info(
        f"Research completed: {result['stop_reason']}",
        extra={"user": user.sub, "action": "research_complete"},
    )
    return result


@app.get("/api/research/health")
async def research_health():
    """Health check — no auth required."""
    registry = get_registry()
    health = registry.check_health()
    return {
        "status": "ok",
        "agents": health,
        "registry": registry.to_dict(),
    }


# ==================== AGENT REGISTRY ENDPOINTS ====================

@app.get("/api/agents")
async def list_agents(user: TokenData = Depends(require_permission("view"))):
    """List all registered agents."""
    registry = get_registry()
    return {"agents": [a.to_dict() for a in registry.list_all()]}


@app.get("/api/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    user: TokenData = Depends(require_permission("view")),
):
    """Get agent details by ID."""
    registry = get_registry()
    agent = registry.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent.to_dict()


@app.post("/api/agents", status_code=201)
async def register_agent(
    req: AgentRegisterRequest,
    user: TokenData = Depends(require_permission("manage_agents")),
):
    """Register a new agent (admin only)."""
    registry = get_registry()
    agent = registry.register(
        agent_id=req.agent_id,
        name=req.name,
        description=req.description,
        tools=req.tools,
        capabilities=req.capabilities,
    )
    logger.info(f"Agent registered: {req.agent_id}", extra={"user": user.sub, "agent": req.agent_id})
    return agent.to_dict()


@app.delete("/api/agents/{agent_id}")
async def deregister_agent(
    agent_id: str,
    user: TokenData = Depends(require_permission("manage_agents")),
):
    """Deregister an agent (admin only)."""
    registry = get_registry()
    if not registry.deregister(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    logger.info(f"Agent deregistered: {agent_id}", extra={"user": user.sub, "agent": agent_id})
    return {"status": "deregistered", "agent_id": agent_id}


@app.post("/api/agents/{agent_id}/heartbeat")
async def agent_heartbeat(
    agent_id: str,
    user: TokenData = Depends(get_current_user),
):
    """Agent heartbeat — agents call this to stay healthy."""
    registry = get_registry()
    if not registry.heartbeat(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found or deregistered")
    return {"status": "ok", "agent_id": agent_id}


@app.get("/api/agents/discover/{capability}")
async def discover_agents(
    capability: str,
    user: TokenData = Depends(require_permission("view")),
):
    """Discover agents by capability."""
    registry = get_registry()
    agents = registry.discover_by_capability(capability)
    return {"capability": capability, "agents": [a.to_dict() for a in agents]}


# ==================== AGENT IDENTITY ENDPOINTS ====================

@app.post("/api/agent-identities", status_code=201)
async def create_agent_identity(
    req: AgentIdentityRequest,
    user: TokenData = Depends(require_permission("manage_agents")),
):
    """Create a service account JWT for an agent (admin only)."""
    role = Role(req.role)
    token = register_agent_identity(req.agent_id, req.agent_name, role)
    logger.info(
        f"Agent identity created: {req.agent_id}",
        extra={"user": user.sub, "agent": req.agent_id},
    )
    return {"agent_id": req.agent_id, "token": token, "role": role.value}


@app.get("/api/agent-identities")
async def list_identities(
    user: TokenData = Depends(require_permission("manage_agents")),
):
    """List all agent identities (admin only)."""
    return {"identities": [i.model_dump() for i in list_agent_identities()]}
