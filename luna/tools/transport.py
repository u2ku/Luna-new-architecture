"""Tool-call transports.

The tool framework (registry, specs, executor, receipts, per-turn budget) is
provider-agnostic. What differs by provider is the **transport**: how a tool
call is read out of a model response and how a bounded result is fed back.

Two transports, selected by the provider's :attr:`supports_native_tools` flag:

* :class:`NativeTransport` — for providers that implement OpenAI-style
  function-calling (the ``tools`` field on the wire and ``tool_calls`` in the
  response). Structured, provider-supported.
* :class:`PromptJsonTransport` — for providers (notably **whooshd**) that do
  **not** support native function-calling. The tool schemas are put in the
  system prompt; the model emits a tool call in a strict ```` ```tool_call ````
  sentinel block in its text; the runtime parses only that sentinel, executes
  through the same executor (same receipts, same limits), and feeds the result
  back as a ```` ```tool_result ```` block. Natural prose answers flow
  untouched.

Safety: only the exact ```` ```tool_call ```` sentinel executes. Generic
```` ```json ```` blocks and inline JSON are ordinary content — the rule that
tools are not parsed from ordinary prose/code blocks still holds.

whooshd accepts ``tools`` on the wire but cannot consume them and 502s when the
model returns ``content: null`` (the shape a tool-call attempt produces). The
prompt transport therefore sends ``tools=()`` — removing that trigger — and
carries the protocol in the prompt instead.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from luna.models.base import Message, ModelResponse, ToolSpec as ModelToolSpec


# ---------------------------------------------------------------------------
# Parsed extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedCall:
    """One tool call pulled out of a model response."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Extraction:
    """Result of parsing one model response for tool calls.

    Exactly one of these is meaningful:

    * ``malformed`` is set — a ```` ```tool_call ```` sentinel was present but
      the JSON/shape was bad; the loop sends a repair message and re-prompts.
    * ``calls`` is non-empty — execute the calls (prompt: the first only;
      native: all, up to the remaining budget).
    * neither — the response is the final answer (the content, as-is).
    """

    calls: list[ExtractedCall] = field(default_factory=list)
    malformed: str | None = None
    has_block: bool = False


# ---------------------------------------------------------------------------
# Sentinel parser
# ---------------------------------------------------------------------------

#: Matches a fenced block whose info string is exactly ``tool_call``,
#: capturing everything (incl. nested braces, multiline JSON) up to the
#: closing fence. Non-greedy so the *first* block wins.
_BLOCK_RE = re.compile(r"```tool_call\s*(.*?)\s*```", re.DOTALL)
_SENTINEL = "```tool_call"


def parse_content(content: str, *, index: int = 0) -> Extraction:
    """Parse model ``content`` for a ```` ```tool_call ```` block.

    One call per turn (first block only). Returns a structured
    :class:`Extraction`; never raises — malformed JSON becomes a
    ``malformed`` extraction the loop can repair.
    """
    if not content or _SENTINEL not in content:
        return Extraction(calls=[], malformed=None, has_block=False)

    match = _BLOCK_RE.search(content)
    if not match:
        # Sentinel present but no closed block (truncated mid-call).
        return Extraction(
            calls=[],
            malformed="a ```tool_call block was opened but not closed",
            has_block=True,
        )

    raw = match.group(1).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return Extraction(
            calls=[],
            malformed=f"invalid JSON in tool_call block: {exc.msg}",
            has_block=True,
        )

    if not isinstance(obj, dict) or "tool" not in obj or "arguments" not in obj:
        return Extraction(
            calls=[],
            malformed="tool_call block must be an object with 'tool' and 'arguments'",
            has_block=True,
        )

    tool = obj["tool"]
    args = obj.get("arguments") or {}
    if not isinstance(tool, str) or not isinstance(args, dict):
        return Extraction(
            calls=[],
            malformed="'tool' must be a string and 'arguments' an object",
            has_block=True,
        )

    call = ExtractedCall(
        call_id=f"prompt-{index}", name=tool, arguments=dict(args)
    )
    return Extraction(calls=[call], malformed=None, has_block=True)


# ---------------------------------------------------------------------------
# Tool protocol prompt (prompt-JSON transport)
# ---------------------------------------------------------------------------


def tool_protocol_prompt(specs: list[ModelToolSpec]) -> str:
    """Build the tool-calling protocol block for the system prompt.

    Lists each enabled tool's name, description, and a compact JSON schema so
    the model can emit a valid ```` ```tool_call ```` block.
    """
    lines: list[str] = [
        "You have access to archive tools. Use them when you need historical context.",
        "",
        "To call a tool, emit exactly ONE fenced block like this and then stop:",
        "```tool_call",
        '{"tool": "<tool_name>", "arguments": { ... }}',
        "```",
        "",
        "Rules:",
        "- Call at most ONE tool per turn. Wait for the result before continuing.",
        "- `arguments` must match the tool's schema. Use only the properties listed.",
        "- Never invent artifact_ids; only use ids returned in a ```tool_result.",
        "- After you receive a ```tool_result, either call another tool or answer.",
        "- If you do not need a tool, answer the user directly in prose.",
        "- Do not wrap tool calls in any other code block; only ```tool_call executes.",
        "- NEVER return an empty reply. You must always either emit a ```tool_call "
        "block or write a prose answer.",
        "",
        "Example — calling search_web:",
        "```tool_call",
        '{"tool": "search_web", "arguments": {"query": "example query", "limit": 5}}',
        "```",
        "",
        "Available tools:",
    ]
    for spec in specs:
        lines.append(f"\n### {spec.name}")
        if spec.description:
            lines.append(spec.description)
        lines.append("arguments:")
        params = spec.parameters if isinstance(spec.parameters, Mapping) else {}
        props = params.get("properties") or {}
        required = params.get("required") or []
        if not props:
            lines.append("  (none)")
        for name, schema in props.items():
            mark = " (required)" if name in required else ""
            desc = schema.get("description", "") if isinstance(schema, Mapping) else ""
            t = schema.get("type", "any") if isinstance(schema, Mapping) else "any"
            lines.append(f"  - {name}: {t}{mark}  {desc}".rstrip())
    lines.append("")
    lines.append("```tool_result blocks are returned to you by the runtime; do not emit them yourself.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class ToolTransport(ABC):
    """How the turn loop reads tool calls and writes tool results."""

    #: Whether ``tools`` should be sent on the wire (native) or kept in the
    #: prompt (prompt-JSON).
    wants_tools_on_wire: bool = True

    @abstractmethod
    def augment_system_prompt(
        self, system_prompt: str, specs: list[ModelToolSpec]
    ) -> str:
        """Return the system prompt, optionally augmented with the protocol."""

    @abstractmethod
    def extract(self, response: ModelResponse, *, call_index: int = 0) -> Extraction:
        """Pull tool calls (or a malformed signal) out of one response."""

    @abstractmethod
    def assistant_message(self, response: ModelResponse) -> Message:
        """The model's own turn, for the conversation history."""

    @abstractmethod
    def tool_result_message(self, call: ExtractedCall, bounded_payload: str) -> Message:
        """A message carrying a bounded tool result back to the model."""

    @abstractmethod
    def repair_message(self, reason: str) -> Message:
        """Message sent when a tool call was malformed, asking for a re-emit."""

    @abstractmethod
    def force_final_message(self) -> Message | None:
        """Optional message to append when the call budget is spent."""

    def empty_reply_nudge(self) -> Message:
        """Message sent when the model returned an empty reply.

        Coaxes a re-emit instead of failing the turn. Subclasses override
        to tailor the instruction (e.g. the prompt-JSON transport reminds
        the model of the ```` ```tool_call ```` sentinel).
        """
        return Message(
            role="user",
            content=(
                "Your previous reply was empty. Answer the user directly, or "
                "call a tool."
            ),
        )


class NativeTransport(ToolTransport):
    """OpenAI-style structured function-calling."""

    wants_tools_on_wire = True

    def augment_system_prompt(self, system_prompt, specs):
        return system_prompt

    def extract(self, response, *, call_index=0):
        calls = [
            ExtractedCall(
                call_id=tc.id,
                name=tc.name,
                arguments=dict(tc.arguments) if isinstance(tc.arguments, Mapping) else {},
            )
            for tc in (response.tool_calls or ())
        ]
        return Extraction(calls=calls, malformed=None, has_block=bool(calls))

    def assistant_message(self, response):
        return Message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls,
        )

    def tool_result_message(self, call, bounded_payload):
        return Message(
            role="tool",
            tool_call_id=call.call_id,
            content=bounded_payload,
        )

    def repair_message(self, reason):
        # Native transport never produces a malformed extraction; the
        # provider validated the structure. Return a no-op user message
        # just in case a subclass routes here.
        return Message(role="user", content=f"(tool call issue: {reason})")

    def force_final_message(self):
        return None


class PromptJsonTransport(ToolTransport):
    """Sentinel-based JSON tool calls in the model's text content.

    Used for providers (whooshd) that do not support native function-calling.
    Tools are described in the system prompt; the model emits a
    ```` ```tool_call ```` block; the runtime parses it, executes through the
    shared executor, and feeds results back as ```` ```tool_result ```` blocks.
    """

    wants_tools_on_wire = False

    def augment_system_prompt(self, system_prompt, specs):
        if not specs:
            return system_prompt
        return tool_protocol_prompt(specs) + "\n\n---\n\n" + system_prompt

    def extract(self, response, *, call_index=0):
        return parse_content(response.content or "", index=call_index)

    def assistant_message(self, response):
        # Keep the full content (including the ```tool_call block) so the
        # model sees its own prior call on the next turn.
        return Message(role="assistant", content=response.content or "")

    def tool_result_message(self, call, bounded_payload):
        return Message(
            role="user",
            content=(
                f"```tool_result\n{bounded_payload}\n```\n\n"
                "(Tool result above. Call another tool with a ```tool_call "
                "block, or give your final answer to the user.)"
            ),
        )

    def repair_message(self, reason):
        return Message(
            role="user",
            content=(
                f"Your previous ```tool_call block was not usable: {reason}. "
                "Re-emit a single valid ```tool_call block with a JSON object "
                "containing \"tool\" and \"arguments\", or answer the user directly."
            ),
        )

    def force_final_message(self):
        return Message(
            role="user",
            content=(
                "You have used your tool budget for this turn. Answer the user "
                "now; do not emit any further ```tool_call blocks."
            ),
        )

    def empty_reply_nudge(self) -> Message:
        return Message(
            role="user",
            content=(
                "Your previous reply was empty. Either emit a single "
                "```tool_call block to call a tool, or answer the user directly "
                "in prose. Do not return an empty reply."
            ),
        )


def select_transport(provider: Any) -> ToolTransport:
    """Pick the transport for a provider by its capability flag."""
    if getattr(provider, "supports_native_tools", True):
        return NativeTransport()
    return PromptJsonTransport()
