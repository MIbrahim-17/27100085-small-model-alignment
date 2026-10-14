import torch
import torch.nn.functional as F


def format_prompt(instruction):
    return "### Instruction:\n" + instruction.strip() + "\n\n### Response:\n"


def build_sft_batch(tokenizer, prompts, responses, max_length):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        pad_id = 0

    all_input_ids = []
    all_labels = []
    for i in range(len(prompts)):
        prompt_ids = list(tokenizer.encode(prompts[i], add_special_tokens=False))
        response_ids = list(tokenizer.encode(responses[i], add_special_tokens=False))
        if eos_id is not None:
            response_ids = response_ids + [eos_id]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    longest = 0
    for input_ids in all_input_ids:
        if len(input_ids) > longest:
            longest = len(input_ids)

    all_attention_masks = []
    for i in range(len(all_input_ids)):
        real_length = len(all_input_ids[i])
        pad_length = longest - real_length
        all_attention_masks.append([1] * real_length + [0] * pad_length)
        all_input_ids[i] = all_input_ids[i] + [pad_id] * pad_length
        all_labels[i] = all_labels[i] + [-100] * pad_length

    batch = {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }
    return batch


def compute_batch_logprobs(model, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    if logits.dtype == torch.float16:
        logits = logits.float()
    targets = labels[:, 1:]

    vocab_size = logits.shape[-1]
    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    token_losses = token_losses.reshape(targets.shape)
    logprobs = -token_losses.sum(dim=1)
    return logprobs


def count_response_tokens(batch):
    targets = batch["labels"][:, 1:]
    return (targets != -100).sum()


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return compute_batch_logprobs(model, batch)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    logprobs = compute_batch_logprobs(model, batch)
    token_count = count_response_tokens(batch).clamp(min=1)
    loss = -logprobs.sum() / token_count.to(logprobs.device)
    return loss


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen_batch = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    losses = -F.logsigmoid(beta * (policy_gap - reference_gap))
    return losses.mean()


def compute_dpo_logps(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)

    policy_chosen_logps = compute_batch_logprobs(policy_model, batch["chosen"])
    policy_rejected_logps = compute_batch_logprobs(policy_model, batch["rejected"])

    with torch.no_grad():
        reference_chosen_logps = compute_batch_logprobs(reference_model, batch["chosen"])
        reference_rejected_logps = compute_batch_logprobs(reference_model, batch["rejected"])

    reference_chosen_logps = reference_chosen_logps.to(policy_chosen_logps.device)
    reference_rejected_logps = reference_rejected_logps.to(policy_chosen_logps.device)

    return policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps = compute_dpo_logps(
        policy_model,
        reference_model,
        tokenizer,
        prompts,
        chosen_responses,
        rejected_responses,
        max_length,
    )
    loss = compute_dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )
    return loss
