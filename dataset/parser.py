"""Fable-5 Trace Parser and Normalizer for SFT and Qwen tokenization."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def merge_consecutive_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive messages of the same role (e.g. multiple consecutive user turns).
    Preserves tool_calls and reasoning_content where applicable.
    """
    if not messages:
        return []

    merged = []
    for msg in messages:
        if not merged:
            merged.append(dict(msg))
            continue

        prev = merged[-1]
        same_role = (prev.get("role") == msg.get("role"))

        # Merge consecutive user messages
        if same_role and msg.get("role") == "user":
            prev_content = prev.get("content", "").strip()
            new_content = msg.get("content", "").strip()
            if prev_content and new_content:
                prev["content"] = f"{prev_content}\n\n{new_content}"
            elif new_content:
                prev["content"] = new_content
        # Merge consecutive system messages
        elif same_role and msg.get("role") == "system":
            prev_content = prev.get("content", "").strip()
            new_content = msg.get("content", "").strip()
            if prev_content and new_content:
                prev["content"] = f"{prev_content}\n\n{new_content}"
            elif new_content:
                prev["content"] = new_content
        else:
            merged.append(dict(msg))

    return merged


def parse_context_into_messages(
    context_str: str,
    fallback_user_prompt: Optional[str] = "Continue the task with the current state.",
    merge_consecutive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Parse a Fable-5 context transcript into structured chat messages.
    
    Recognizes:
      - USER: <prompt>
      - ASSISTANT (tool call) <tool_name> input=<json_or_text>
      - TOOL RESULT: <result>
      - ASSISTANT (message): <text>
      - ASSISTANT: <text>
      - SYSTEM: <text>
    """
    if not context_str:
        if fallback_user_prompt:
            return [{"role": "user", "content": fallback_user_prompt}]
        return []

    header_pattern = re.compile(
        r"^(USER:|ASSISTANT \(tool call\)|TOOL RESULT:|ASSISTANT \(message\):|ASSISTANT:|SYSTEM:)",
        re.MULTILINE,
    )

    matches = list(header_pattern.finditer(context_str))
    if not matches:
        return [{"role": "user", "content": context_str.strip()}]

    messages = []
    for i, match in enumerate(matches):
        header = match.group(1)
        start_idx = match.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(context_str)
        chunk = context_str[start_idx:end_idx].strip()

        if header == "USER:":
            if chunk:  # Avoid empty user chunks
                messages.append({"role": "user", "content": chunk})
        elif header == "SYSTEM:":
            if chunk:
                messages.append({"role": "system", "content": chunk})
        elif header in ("ASSISTANT (message):", "ASSISTANT:"):
            messages.append({"role": "assistant", "content": chunk})
        elif header == "ASSISTANT (tool call)":
            m_tool = re.match(r"^\s*([\w\-]+)\s+input=(.*)", chunk, re.DOTALL)
            if m_tool:
                tool_name = m_tool.group(1)
                tool_input_raw = m_tool.group(2).strip()
                try:
                    tool_args = json.loads(tool_input_raw)
                except Exception:
                    tool_args = {"raw": tool_input_raw}
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args,
                        }
                    }]
                })
            else:
                messages.append({"role": "assistant", "content": chunk})
        elif header == "TOOL RESULT:":
            messages.append({"role": "tool", "content": chunk if chunk else "[No output]"})

    # Check if there is at least one user message
    has_user = any(m.get("role") == "user" for m in messages)
    if not has_user and fallback_user_prompt:
        messages.insert(0, {"role": "user", "content": fallback_user_prompt})

    if merge_consecutive:
        messages = merge_consecutive_messages(messages)

    return messages


def build_target_assistant_turn(record: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the ground truth assistant target turn from a Fable-5 record."""
    cot = record.get("cot", "")
    out_type = record.get("output_type", "text")
    out_data = record.get("output", {})

    target: Dict[str, Any] = {
        "role": "assistant",
        "reasoning_content": cot,
    }

    if out_type == "tool_use":
        target["content"] = ""
        tool_name = out_data.get("tool", "unknown_tool")
        tool_input = out_data.get("input", {})
        target["tool_calls"] = [{
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": tool_input,
            }
        }]
    else:
        text_content = out_data.get("text", "") if isinstance(out_data, dict) else str(out_data)
        target["content"] = text_content

    return target


def parse_fable_record_to_messages(
    record: Dict[str, Any],
    fallback_user_prompt: Optional[str] = "Continue the task with the current state.",
    merge_consecutive: bool = True,
) -> List[Dict[str, Any]]:
    """Convert a Fable-5 record into a full list of messages (Context + Target Assistant Turn)."""
    context_msgs = parse_context_into_messages(
        record.get("context", ""),
        fallback_user_prompt=fallback_user_prompt,
        merge_consecutive=merge_consecutive,
    )
    target_turn = build_target_assistant_turn(record)
    return context_msgs + [target_turn]


def extract_tools_from_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract unique tool schemas observed in the dataset for system prompt formatting."""
    tool_map: Dict[str, Dict[str, Any]] = {}

    for r in records:
        if r.get("output_type") == "tool_use":
            out = r.get("output", {})
            t_name = out.get("tool")
            t_input = out.get("input", {})
            if t_name and t_name not in tool_map:
                properties = {}
                if isinstance(t_input, dict):
                    for k, v in t_input.items():
                        v_type = "string"
                        if isinstance(v, bool):
                            v_type = "boolean"
                        elif isinstance(v, int):
                            v_type = "integer"
                        elif isinstance(v, float):
                            v_type = "number"
                        elif isinstance(v, dict):
                            v_type = "object"
                        elif isinstance(v, list):
                            v_type = "array"
                        properties[k] = {"type": v_type, "description": f"Parameter {k}"}
                tool_map[t_name] = {
                    "type": "function",
                    "function": {
                        "name": t_name,
                        "description": f"Tool {t_name}",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                        },
                    },
                }

    return list(tool_map.values())
