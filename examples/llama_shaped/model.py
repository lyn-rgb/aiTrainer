"""A Llama-shaped causal LM, written the way HuggingFace ships one.

Nothing here is framework-specific: this is an ordinary ``nn.Module`` and the
guide's point is that it stays one.  What it reproduces is the *structure* that
decides which parallel axes are a configuration change and which need the model
to say something:

* projections are named ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` and
  ``gate_proj`` / ``up_proj`` / ``down_proj``, which is what the default TP plan
  matches on -- a model that names them something else is refused, not silently
  replicated;
* the norms are ``nn.RMSNorm``.  Sequence parallelism matches norms by *type*,
  so this works with no naming convention at all;
* **the layers are nested one level down**, inside ``self.model.layers``, with
  ``self.lm_head`` beside ``self.model`` rather than inside it.  That is HF's
  arrangement, and it is the one thing a pipeline split cannot infer: see
  :func:`execution_order` at the bottom.

Run ``train.py`` for the guide's worked examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LlamaShapedConfig:
    """Small enough to train on a laptop, shaped like the real thing.

    Deliberately tiny -- ~30k parameters.  Every process in a multi-rank run
    loads its own copy of torch AND its own copy of this model, so the model is
    not where the memory goes; keeping it small is what keeps a 4-rank run
    cheap enough to be unremarkable on a laptop.  The shape is what matters
    here, not the size: names the TP plan matches, ``nn.RMSNorm`` norms, and
    layers nested one level down.
    """

    vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 4
    num_heads: int = 4
    max_position: int = 32

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class Attention(nn.Module):
    """Multi-head attention, and the one place this model is tensor-parallel aware.

    See :func:`_per_head` for why, and for what it costs.  Everything else in
    this file is written as if the model were running on one device.
    """
    def __init__(self, config: LlamaShapedConfig) -> None:
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.head_dim
        # bias=False, as Llama has it.  A row projection's bias would be
        # replicated under TP; leaving it out keeps this model's parameter set
        # identical to the real one.
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        shape = (batch, length, self.heads, self.head_dim)

        def split(projection):
            return projection(hidden).view(shape).transpose(1, 2)

        query, key, value = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        attended = _per_head(query, key, value)
        # Explicit width, not -1: under tensor parallelism this reshape merges
        # the head axis back into the feature axis, and an inferred dimension
        # gives DTensor nothing to check the sharding against.
        merged = attended.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)
        return self.o_proj(merged)


def _per_head(query, key, value):
    """``scaled_dot_product_attention`` over heads that may be split across ranks.

    Under tensor parallelism a column projection hands back a ``DTensor``
    sharded on the **head** axis -- each rank holds two of this model's four
    heads, whole.  That is the arrangement the sharding rule wants: attention is
    independent per head, so a rank holding complete heads can compute them
    exactly.

    DTensor cannot be told that.  Measured at ``tp_size=2`` with the placements
    preserved::

        NotImplementedError: Operator
        aten._scaled_dot_product_flash_attention_for_cpu.default does not have a
        sharding strategy registered.

    so this is the local round-trip -- drop to the local tensor, compute, and
    put the placement back.  It is a no-op when the model is not sharded, and it
    needs no knowledge of ``tp_size``: the mesh and the placement are read off
    the tensor the projection produced, so the same code runs at ``tp_size=1``,
    2 or 8.  A CUDA build registers a strategy for the flash op and may not need
    this; on CPU/Gloo it does.

    This is the one place the guide's example is parallelism-aware, and it is
    worth being precise about why it must be: an op with no sharding strategy is
    not something a framework can paper over, because only the model knows that
    computing it per rank is the *same* computation rather than a wrong one.
    """
    if not hasattr(query, "to_local"):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    from torch.distributed.tensor import DTensor, Shard

    # The sharded axis here is the HEAD axis at dim 1: the projections emit
    # (batch, length, heads, head_dim) sharded at dim 2, and ``Attention.forward``
    # transposes 1 and 2 before calling this.  Getting that number wrong would
    # not fail -- it would label the sequence axis as sharded and produce a
    # DTensor that describes a tensor nobody computed.
    attended = torch.nn.functional.scaled_dot_product_attention(
        query.to_local(), key.to_local(), value.to_local())
    return DTensor.from_local(attended, query.device_mesh, [Shard(1)])


class MLP(nn.Module):
    def __init__(self, config: LlamaShapedConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Block(nn.Module):
    """The standard pre-norm block, with HF's module names."""

    def __init__(self, config: LlamaShapedConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, hidden):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class LlamaShapedModel(nn.Module):
    """The trunk: embedding, layer stack, final norm.  ``model`` in HF's naming."""

    def __init__(self, config: LlamaShapedConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class LlamaShapedForCausalLM(nn.Module):
    """``model`` + ``lm_head`` as *siblings* -- HF's shape, and the one PP needs
    to be told about.
    """

    def __init__(self, config: LlamaShapedConfig | None = None) -> None:
        super().__init__()
        self.config = config or LlamaShapedConfig()
        self.model = LlamaShapedModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))

    def loss(self, output, batch):
        return causal_lm_loss(output, batch)


def causal_lm_loss(output, batch):
    """The ``loss_fn`` for every strategy in the guide.

    A plain function rather than the method above, because a pipeline run has
    no single model object to hang it on: ``parallelize`` returns the stage this
    rank owns, and the loss is the same expression on every stage.
    """
    labels = batch[1]
    return torch.nn.functional.cross_entropy(
        output[:, :-1].reshape(-1, output.shape[-1]),
        labels[:, 1:].reshape(-1),
    )


def execution_order(model: LlamaShapedForCausalLM):
    """The ordered units a pipeline split divides, including the glue.

    A ``PipelineSequence`` runs each unit by calling it, so the units have to be
    callables that compose to this model's ``forward``: embedding, then each
    layer, then the final norm, then the head.  ``model.named_children()``
    returns ``model`` and ``lm_head`` -- the trunk and the head -- so without
    this the whole transformer lands on stage 0.

    Names must be dot-free (``PipelineSequence`` registers each unit with
    ``add_module``, which rejects ``"."``).  They become the stage's checkpoint
    keys, so keep them stable across runs; their leaf names are what the TP plan
    matches, and ``layer_0.self_attn.q_proj`` still matches ``q_proj``.
    """
    return [
        ("embed_tokens", model.model.embed_tokens),
        *[(f"layer_{index}", layer) for index, layer in enumerate(model.model.layers)],
        ("norm", model.model.norm),
        ("lm_head", model.lm_head),
    ]


def build(seed: int = 0, config: LlamaShapedConfig | None = None) -> LlamaShapedForCausalLM:
    torch.manual_seed(seed)
    return LlamaShapedForCausalLM(config)


def batches(seed: int = 0, steps: int = 4, batch: int = 4, length: int = 16,
            config: LlamaShapedConfig | None = None):
    config = config or LlamaShapedConfig()
    torch.manual_seed(seed + 1)
    return [(torch.randint(0, config.vocab_size, (batch, length)),
             torch.randint(0, config.vocab_size, (batch, length)))
            for _ in range(steps)]
