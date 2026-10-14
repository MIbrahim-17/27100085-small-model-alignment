import copy

import torch

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoEosTokenizer(ToyTokenizer):
    eos_token = None
    eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[character] for character in text]


def manual_logprob(model, tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + [tokenizer.eos_token_id]
    ids = (prompt_ids + response_ids)[:max_length]
    log_probs = torch.log_softmax(model.logit_bias, dim=0)
    total = torch.tensor(0.0)
    for position in range(len(prompt_ids), len(ids)):
        if position == 0:
            continue
        total = total + log_probs[ids[position]]
    return total


def test_truncation_removes_eos_and_keeps_max_length():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["abc: "], ["abcdefghij"], max_length=8)
    assert batch["input_ids"].shape == (1, 8)
    assert batch["labels"][0, :5].eq(-100).all()
    assert batch["labels"][0, 5:].ne(-100).all()
    assert tokenizer.eos_token_id not in batch["input_ids"][0].tolist()


def test_eos_is_kept_when_sequence_fits_exactly():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["abc: "], ["ab"], max_length=8)
    assert batch["input_ids"].shape == (1, 8)
    assert batch["input_ids"][0, -1] == tokenizer.eos_token_id
    assert batch["labels"][0, -1] == tokenizer.eos_token_id


def test_prompt_longer_than_max_length_has_no_response_tokens():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tokenizer, ["a very long prompt: "], ["ok"], max_length=6)
    assert batch["input_ids"].shape == (1, 6)
    assert batch["labels"].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tokenizer, ["a very long prompt: "], ["ok"], max_length=6)
    assert logps.shape == (1,)
    assert logps[0] == 0
    loss = adapter.compute_sft_loss(model, tokenizer, ["a very long prompt: "], ["ok"], max_length=6)
    assert torch.isfinite(loss)


def test_tokenizer_without_eos_does_not_append_anything():
    tokenizer = NoEosTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q: "], ["yes"], max_length=16)
    assert batch["input_ids"].shape == (1, 6)
    assert batch["labels"][0, 3:].tolist() == tokenizer.encode("yes")


def test_mixed_lengths_keep_row_order_and_match_manual_values():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.0)
    prompts = ["Q: ", "A much longer question here: ", "x", "Mid: "]
    responses = ["a long response with many tokens", "b", "medium text", "z"]
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=24)
    assert logps.shape == (4,)
    for row in range(4):
        expected = manual_logprob(model, tokenizer, prompts[row], responses[row], 24)
        torch.testing.assert_close(logps[row], expected, rtol=1e-5, atol=1e-5)

    single = adapter.compute_response_logprobs(model, tokenizer, prompts[2:3], responses[2:3], max_length=24)
    torch.testing.assert_close(single[0], logps[2], rtol=1e-5, atol=1e-5)


def test_padding_does_not_change_logprobs_and_masks_match():
    tokenizer = ToyTokenizer()
    batch = adapter.build_sft_batch(tokenizer, ["Q: ", "Question: "], ["a", "a longer answer"], max_length=64)
    pad_positions = batch["attention_mask"].eq(0)
    assert pad_positions.any()
    assert batch["labels"][pad_positions].eq(-100).all()
    assert batch["input_ids"][pad_positions].eq(tokenizer.pad_token_id).all()
    assert batch["input_ids"].shape[1] <= 64


def test_dpo_loss_from_logps_matches_formula():
    policy_chosen = torch.tensor([-1.0, -4.0, -2.5])
    policy_rejected = torch.tensor([-3.0, -1.0, -2.5])
    reference_chosen = torch.tensor([-2.0, -2.0, -3.0])
    reference_rejected = torch.tensor([-2.5, -3.0, -1.0])
    beta = 0.3
    difference = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    expected = -torch.log(torch.sigmoid(beta * difference)).mean()
    loss = adapter.compute_dpo_loss_from_logps(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    torch.testing.assert_close(loss, expected)


def test_dpo_loss_matches_logprob_pieces_and_identical_models_give_log_two():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.0)
    reference = ToyCausalLM(tokenizer.vocab_size, bias_scale=-0.5)
    prompts = ["Pick: ", "Choose wisely: "]
    chosen = ["good answer", "yes"]
    rejected = ["bad", "no thanks"]
    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta=0.1, max_length=32)
    expected = adapter.compute_dpo_loss_from_logps(
        adapter.compute_response_logprobs(policy, tokenizer, prompts, chosen, 32),
        adapter.compute_response_logprobs(policy, tokenizer, prompts, rejected, 32),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, chosen, 32),
        adapter.compute_response_logprobs(reference, tokenizer, prompts, rejected, 32),
        0.1,
    )
    torch.testing.assert_close(loss, expected)

    same = copy.deepcopy(policy)
    tied_loss = adapter.compute_dpo_loss(policy, same, tokenizer, prompts, chosen, rejected, beta=0.1, max_length=32)
    torch.testing.assert_close(tied_loss, torch.log(torch.tensor(2.0)))


def test_small_optimization_steps_reduce_both_losses():
    tokenizer = ToyTokenizer()
    prompts = ["Pick: ", "Choose: "]
    chosen = ["aaa", "abab"]
    rejected = ["zzz", "zyzy"]

    sft_model = ToyCausalLM(tokenizer.vocab_size)
    optimizer = torch.optim.SGD(sft_model.parameters(), lr=0.5)
    first_loss = adapter.compute_sft_loss(sft_model, tokenizer, prompts, chosen, max_length=16)
    for step in range(5):
        optimizer.zero_grad()
        loss = adapter.compute_sft_loss(sft_model, tokenizer, prompts, chosen, max_length=16)
        loss.backward()
        optimizer.step()
    last_loss = adapter.compute_sft_loss(sft_model, tokenizer, prompts, chosen, max_length=16)
    assert last_loss < first_loss

    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    reference_before = reference.logit_bias.detach().clone()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    first_loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, 0.5, 16)
    for step in range(5):
        optimizer.zero_grad()
        loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, 0.5, 16)
        loss.backward()
        optimizer.step()
    last_loss = adapter.compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, 0.5, 16)
    assert last_loss < first_loss
    assert torch.equal(reference.logit_bias.detach(), reference_before)
    assert reference.logit_bias.grad is None


def test_double_precision_model_keeps_double_precision():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=1.0).double()
    logps = adapter.compute_response_logprobs(model, tokenizer, ["q: "], ["abc"], 16)
    assert logps.dtype == torch.float64
