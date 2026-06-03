"""Auth routes: exchange an access key for a JWT."""
from fastapi import APIRouter, Depends

from ..schemas import LoginRequest, LoginResponse
from ..auth import authenticate_key, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    username = authenticate_key(body.key)
    token, expires_in = create_access_token(username)
    return LoginResponse(token=token, username=username, expires_in=expires_in)


@router.get("/me")
def me(username: str = Depends(get_current_user)) -> dict:
    """Lets the frontend confirm a stored token is still valid on reload."""
    return {"username": username}
