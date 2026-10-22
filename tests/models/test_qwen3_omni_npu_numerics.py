# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run on Ascend: pytest -q tests/models/test_qwen3_omni_npu_numerics.py."""

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as modeling
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

from verl_omni.models.transformers.qwen3_omni_npu import apply_qwen3_omni_thinker_npu_patch


@pytest.fixture
def install_patch(monkeypatch):
    # Restore global forwards so every reference run uses Transformers' implementation.
    for name in (
        "Qwen3OmniMoeThinkerTextExperts",
        "Qwen3OmniMoeThinkerTextRMSNorm",
        "Qwen3OmniMoeTextRMSNorm",
        "Qwen3OmniMoeThinkerTextMLP",
    ):
        cls = getattr(modeling, name)
        monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", modeling.apply_rotary_pos_emb)
    return lambda: apply_qwen3_omni_thinker_npu_patch(SimpleNamespace(hidden_act="silu"))


@pytest.mark.parametrize("kind", ["experts", "decoder_norm", "final_norm", "mlp"])
def test_forward_and_backward_match_transformers(install_patch, kind):
    torch.manual_seed(7960)
    config = Qwen3OmniMoeTextConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    if kind == "experts":
        reference = modeling.Qwen3OmniMoeThinkerTextExperts(config)
        with torch.no_grad():
            for param in reference.parameters():
                param.normal_(std=0.02)
    elif kind == "mlp":
        reference = modeling.Qwen3OmniMoeThinkerTextMLP(config)
    elif kind == "decoder_norm":
        reference = modeling.Qwen3OmniMoeThinkerTextRMSNorm(128)
    else:
        reference = modeling.Qwen3OmniMoeTextRMSNorm(128)
    reference = reference.to(device="npu", dtype=torch.bfloat16)
    patched = copy.deepcopy(reference)
    inputs = torch.randn(16, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    patched_inputs = inputs.detach().clone().requires_grad_()
    weights = torch.randn(16, 2, device="npu").softmax(-1).detach().requires_grad_()
    patched_weights = weights.detach().clone().requires_grad_()
    # Leave one expert empty and include repeated assignments across tokens.
    indices = torch.tensor([[0, 1], [1, 2]] * 8, device="npu", dtype=torch.int64)
    args = (indices, weights) if kind == "experts" else ()
    expected = reference(inputs, *args)
    expected.float().square().sum().backward()
    install_patch()
    args = (indices, patched_weights) if kind == "experts" else ()
    actual = patched(patched_inputs, *args)
    actual.float().square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1e-4 if kind == "experts" else 2e-2)
    torch.testing.assert_close(patched_inputs.grad, inputs.grad, rtol=3e-2, atol=1e-4)
    assert patched.state_dict().keys() == reference.state_dict().keys()
    for name, param in patched.named_parameters():
        torch.testing.assert_close(param.grad, reference.get_parameter(name).grad, rtol=3e-2, atol=1e-4)
    if kind == "experts":
        torch.testing.assert_close(patched_weights.grad, weights.grad, rtol=3e-2, atol=1e-4)


@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
def test_rope_forward_and_backward_match_transformers(install_patch, unsqueeze_dim):
    torch.manual_seed(7960)
    shape = (2, 4, 8, 128) if unsqueeze_dim == 1 else (2, 8, 4, 128)
    query = torch.randn(shape, device="npu", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(shape, device="npu", dtype=torch.bfloat16, requires_grad=True)
    patched_query = query.detach().clone().requires_grad_()
    patched_key = key.detach().clone().requires_grad_()
    angles = torch.randn(2, 8, 64, device="npu", dtype=torch.bfloat16).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    expected = modeling.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim)
    sum(output.float().square().sum() for output in expected).backward()
    install_patch()
    actual = modeling.apply_rotary_pos_emb(patched_query, patched_key, cos, sin, unsqueeze_dim)
    sum(output.float().square().sum() for output in actual).backward()
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(patched_query.grad, query.grad, rtol=3e-2, atol=1e-4)
    torch.testing.assert_close(patched_key.grad, key.grad, rtol=3e-2, atol=1e-4)
