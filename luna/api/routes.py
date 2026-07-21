"""HTTP routes for the first Luna runtime vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from luna.context.builder import build_recent_messages
from luna.ledger import WorldLedger
from luna.models.base import Message, ModelError, ModelProvider, ModelRequest

# How many recent user/assistant turns the model gets as context.
# Picked to fit comfortably under gemma's 32k context window with
# room for the system prompt, the new turn, and the model's reply.
CONTEXT_TURNS = 24


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    session_id: str | None = None
    source: str = "web"  # platform name: web, slack, google_chat, ...
    sender_id: str | None = None
    sender_name: str | None = None
    # Channel routing — used to build the stream_id. For the web UI
    # these are derived from session_id; for Slack/Google/etc. the
    # adapter fills them in from the platform-native event.
    account_id: str | None = None
    conversation_id: str | None = None
    thread_id: str | None = None
    external_actor_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    stream_id: str
    turn_id: str
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
    # Default actor identity for the human side of web sessions. Slack
    # and other adapters override this from the platform-native user.
    default_user_id: str = "identity:anonymous"
    default_user_display: str = "User"
    # Runtime identity for outbound events. Configurable so the
    # runtime can be deployed as a different agent name.
    agent_id: str = "agent:luna"
    agent_display_name: str = "Luna"

    def _build_stream_id(
        self,
        platform: str,
        account_id: str | None,
        conversation_id: str,
        thread_id: str,
    ) -> str:
        """Canonical stream id: <platform>:<account_id>:<conversation_id>:<thread_id>.

        Empty fields are kept as empty strings (e.g. ``web::session-id:``)
        rather than collapsed — a stable format makes the stream_id
        easy to grep and parse.
        """
        return f"{platform}:{account_id or ''}:{conversation_id}:{thread_id}"

    def _build_event(
        self,
        *,
        type_: str,
        actor: dict,
        source: dict,
        destination: dict,
        stream_id: str,
        turn_id: str,
        payload: dict,
    ) -> dict:
        return {
            "type": type_,
            "actor": actor,
            "source": source,
            "destination": destination,
            "stream_id": stream_id,
            "turn_id": turn_id,
            "payload": payload,
        }

    def complete(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or str(uuid4())
        turn_id = str(uuid4())
        platform = request.source

        # Web platform: session_id is the conversation and thread.
        # Other platforms: account/conversation/thread come from the
        # adapter via the ChatRequest fields.
        conversation_id = request.conversation_id or session_id
        thread_id = request.thread_id or session_id
        account_id = request.account_id  # may be None for web

        stream_id = self._build_stream_id(
            platform, account_id, conversation_id, thread_id
        )

        # Read recent context for THIS stream BEFORE writing the new
        # user_event. Different streams must never share context.
        recent = build_recent_messages(stream_id=stream_id, limit=CONTEXT_TURNS)
        context_messages = tuple(
            Message(
                role="user" if e.is_user else "assistant",
                content=e.text,
            )
            for e in recent
            if e.text
        )

        user_actor = {
            "id": request.sender_id or self.default_user_id,
            "type": "human",
        }
        if request.sender_name:
            user_actor["display_name"] = request.sender_name
        elif self.default_user_display:
            user_actor["display_name"] = self.default_user_display

        user_source: dict = {"platform": platform, "adapter": "fastapi"}
        if account_id:
            user_source["account_id"] = account_id
        user_source["conversation_id"] = conversation_id
        if thread_id:
            user_source["thread_id"] = thread_id
        if request.external_actor_id:
            user_source["external_actor_id"] = request.external_actor_id

        user_event = self.ledger.append(
            event_type="user_message",
            actor=user_actor,
            source=user_source,
            destination={"platform": "luna-runtime"},
            stream_id=stream_id,
            turn_id=turn_id,
            payload={
                "text": request.text,
                "sender_name": request.sender_name,
            },
        )

        model_request = ModelRequest(
            messages=(
                Message(role="system", content=self.system_prompt),
                *context_messages,
                Message(role="user", content=request.text),
            ),
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata={
                "session_id": session_id,
                "stream_id": stream_id,
                "turn_id": turn_id,
                "user_event_id": user_event["event_id"],
                "context_messages": len(context_messages),
            },
        )

        try:
            model_response = self.provider.complete(model_request)
        except ModelError as exc:
            self.ledger.append(
                event_type="system_event",
                actor={"id": "system:luna-runtime", "type": "system"},
                source={"platform": "luna-runtime"},
                destination={"platform": platform},
                stream_id=stream_id,
                turn_id=turn_id,
                payload={
                    "subtype": "model_call_failed",
                    "provider": self.provider.name,
                    "error": str(exc),
                    "user_event_id": user_event["event_id"],
                },
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        text = model_response.content.strip()
        if not text:
            raise HTTPException(status_code=502, detail="Model returned an empty reply")

        assistant_destination: dict = {"platform": platform, "adapter": "fastapi"}
        if account_id:
            assistant_destination["account_id"] = account_id
        assistant_destination["conversation_id"] = conversation_id
        if thread_id:
            assistant_destination["thread_id"] = thread_id

        assistant_event = self.ledger.append(
            event_type="assistant_message",
            actor={
                "id": self.agent_id,
                "type": "agent",
                "display_name": self.agent_display_name,
            },
            source={"platform": "luna-runtime"},
            destination=assistant_destination,
            stream_id=stream_id,
            turn_id=turn_id,
            payload={
                "text": text,
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
            stream_id=stream_id,
            turn_id=turn_id,
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
