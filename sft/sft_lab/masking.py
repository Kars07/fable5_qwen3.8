"""Assistant loss masking and token role tagging utilities."""

from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizer


def build_sft_labels_and_metadata(
    messages: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 4096,
    assistant_only_loss: bool = True,
) -> Dict[str, Any]:
    """
    Render conversation using chat template, tokenize, assign loss labels and role tags.

    Rules for assistant_only_loss:
    - Non-assistant tokens (system, user, tool responses, prompt headers) receive -100 label.
    - Assistant reasoning (<think>...</think>), tool calls (<tool_call>...</tool_call>), and response text receive token_id label.
    - Assistant terminating EOS token (<|im_end|>) receives token_id label so the model learns when to stop.
    - Truncates sequence cleanly up to max_seq_length.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)

    labels = [-100] * len(input_ids)
    roles = ["system"] * len(input_ids)

    if not assistant_only_loss:
        labels = list(input_ids)
    else:
        # Check if this is a standard step-level conversation ending in an assistant turn
        last_is_assistant = (messages and messages[-1].get("role") == "assistant")
        
        if last_is_assistant:
            # Prompt is everything before the final assistant turn
            prompt_msgs = messages[:-1]
            if prompt_msgs:
                prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                prompt_len = min(len(prompt_ids), len(input_ids))
            else:
                prompt_len = 0

            # Supervise assistant tokens from prompt_len to the end
            for t_idx in range(prompt_len, len(input_ids)):
                labels[t_idx] = input_ids[t_idx]
                roles[t_idx] = "assistant"
                
            for t_idx in range(0, prompt_len):
                roles[t_idx] = "context"
        else:
            # Multi-turn assistant span detection using special token markers
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id is None or im_end_id == tokenizer.unk_token_id:
                im_end_id = tokenizer.eos_token_id

            # Parse assistant blocks directly from token IDs
            in_assistant = False
            for i, tid in enumerate(input_ids):
                # Check for assistant start header
                if tid == im_start_id and i + 1 < len(input_ids):
                    next_tok = tokenizer.decode([input_ids[i + 1]]).strip()
                    if next_tok == "assistant":
                        in_assistant = True
                        roles[i] = "assistant_header"
                        continue

                if in_assistant:
                    roles[i] = "assistant"
                    labels[i] = tid
                    if tid == im_end_id:
                        in_assistant = False
                else:
                    roles[i] = "prompt"
                    labels[i] = -100

    # Truncate sequence cleanly to max_seq_length
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]
        roles = roles[:max_seq_length]

    attention_mask = [1] * len(input_ids)

    return {
        "full_text": full_text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "roles": roles,
    }


def create_token_inspection_table(
    input_ids: List[int],
    labels: List[int],
    roles: List[str],
    tokenizer: PreTrainedTokenizer,
) -> List[Dict[str, Any]]:
    """Generate detailed token inspection rows."""
    table = []
    for idx, (tid, label, role) in enumerate(zip(input_ids, labels, roles)):
        tok_str = tokenizer.decode([tid])
        is_special = tid in tokenizer.all_special_ids
        trained = label != -100
        table.append(
            {
                "idx": idx,
                "token": tok_str,
                "token_id": tid,
                "label": label,
                "trained": trained,
                "role": role,
                "is_special": is_special,
            }
        )
    return table
