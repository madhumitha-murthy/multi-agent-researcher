# platform/auth.py - OAuth2 + JWT authentication and RBAC
#
# Implements:
# - JWT token creation and validation
# - OAuth2 password and client_credentials flows
# - Role-based access control (admin, researcher, viewer)
# - Agent identity via service account JWTs

import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Configuration via environment variables — no hard-coded secrets
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Role(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


# Permission matrix: what each role can do
ROLE_PERMISSIONS = {
    Role.ADMIN: {"research", "view", "manage_agents", "manage_users", "view_logs"},
    Role.RESEARCHER: {"research", "view", "view_logs"},
    Role.VIEWER: {"view"},
}


class TokenData(BaseModel):
    sub: str
    role: Role
    scopes: list[str] = []
    is_agent: bool = False
    exp: Optional[datetime] = None


class User(BaseModel):
    username: str
    hashed_password: str
    role: Role
    disabled: bool = False


class AgentIdentity(BaseModel):
    agent_id: str
    agent_name: str
    role: Role = Role.RESEARCHER
    scopes: list[str] = []


# In-memory user store (replace with DB in production)
_users: dict[str, User] = {
    "admin": User(
        username="admin",
        hashed_password=pwd_context.hash("admin"),
        role=Role.ADMIN,
    ),
    "researcher": User(
        username="researcher",
        hashed_password=pwd_context.hash("researcher"),
        role=Role.RESEARCHER,
    ),
    "viewer": User(
        username="viewer",
        hashed_password=pwd_context.hash("viewer"),
        role=Role.VIEWER,
    ),
}

# Agent service accounts
_agent_identities: dict[str, AgentIdentity] = {}


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def authenticate_user(username: str, password: str) -> Optional[User]:
    user = _users.get(username)
    if not user or user.disabled:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(
    sub: str,
    role: Role,
    scopes: list[str] | None = None,
    is_agent: bool = False,
    expires_delta: timedelta | None = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": sub,
        "role": role.value,
        "scopes": scopes or list(ROLE_PERMISSIONS.get(role, set())),
        "is_agent": is_agent,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(sub: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": sub, "exp": expire, "type": "refresh"},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> TokenData:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return TokenData(
        sub=payload["sub"],
        role=Role(payload["role"]),
        scopes=payload.get("scopes", []),
        is_agent=payload.get("is_agent", False),
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
    )


def validate_token(token: str) -> TokenData:
    """Decode and validate a JWT. Raises JWTError on failure."""
    return decode_token(token)


def has_permission(token_data: TokenData, permission: str) -> bool:
    return permission in token_data.scopes


def register_agent_identity(agent_id: str, agent_name: str, role: Role = Role.RESEARCHER) -> str:
    """Register an agent and return a long-lived service account JWT."""
    scopes = list(ROLE_PERMISSIONS.get(role, set()))
    identity = AgentIdentity(
        agent_id=agent_id, agent_name=agent_name, role=role, scopes=scopes,
    )
    _agent_identities[agent_id] = identity
    return create_access_token(
        sub=f"agent:{agent_id}",
        role=role,
        scopes=scopes,
        is_agent=True,
        expires_delta=timedelta(days=365),
    )


def get_agent_identity(agent_id: str) -> Optional[AgentIdentity]:
    return _agent_identities.get(agent_id)


def list_agent_identities() -> list[AgentIdentity]:
    return list(_agent_identities.values())
