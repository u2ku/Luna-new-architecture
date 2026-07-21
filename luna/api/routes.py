"""HTTP routes for the first Luna runtime vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from luna.ledger import WorldLedger
from luna.models.base import Message, ModelError, ModelProvider, ModelRequest


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    session_id: str | None = None
    source: str = "web"
    sender_id: str | None = None
    sender_name: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    provider: str
    model: str
    user_event_id: str
    assistant_event_id: str


@dataclass
class ChatService:
    provider: ModelProvider
    ledger: WorldLedger
    system_prompt: str
    model_name: str | None = None
    temperature: float | None = 0.3
    max_tokens: int | None = 800

    def complete(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or str(uuid4())

        user_event = self.ledger.append(
            event_type="user_message",
            actor=request.sender_id or "user",
            payload={
                "text": request.text,
                "session_id": session_id,
                "source": request.source,
                "sender_name": request.sender_name,
            },
        )

        model_request = ModelRequest(
            messages=(
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=request.text),
            ),
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata={
                "session_id": session_id,
                "source": request.source,
                "user_event_id": user_event["event_id"],
            },
        )

        try:
            model_response = self.provider.complete(model_request)
        except ModelError as exc:
            self.ledger.append(
                event_type="system_event",
                actor="runtime",
                payload={
                    "subtype": "model_call_failed",
                    "session_id": session_id,
                    "provider": self.provider.name,
                    "error": str(exc),
                    "user_event_id": user_event["event_id"],
                },
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        text = model_response.content.strip()
        if not text:
            raise HTTPException(status_code=502, detail="Model returned an empty reply")

        assistant_event = self.ledger.append(
            event_type="assistant_message",
            actor="luna",
            payload={
                "text": text,
                "session_id": session_id,
                "source": request.source,
                "provider": self.provider.name,
                "model": model_response.model or self.model_name or "",
                "finish_reason": model_response.finish_reason.value,
                "usage": {
                    "prompt_tokens": model_response.usage.prompt_tokens,
                    "completion_tokens": model_response.usage.completion_tokens,
                    "total_tokens": model_response.usage.total_tokens,
                },
                "reply_to_event_id": user_event["event_id"],
            },
        )

        return ChatResponse(
            response=text,
            session_id=session_id,
            provider=self.provider.name,
            model=model_response.model or self.model_name or "",
            user_event_id=user_event["event_id"],
            assistant_event_id=assistant_event["event_id"],
        )


def create_api_router(service: ChatService) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "luna-runtime",
            "provider": service.provider.name,
        }

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        return service.complete(request)

    @router.get("/events")
    def events(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        values = service.ledger.tail(limit=limit)
        return {"events": values, "count": len(values)}

    return router
