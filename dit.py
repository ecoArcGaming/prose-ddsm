import copy
from typing import Optional, Tuple, TypeVar, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
import torch.nn.functional as F
from PoET.poet.models.modules.activation import gelu
from PoET.poet.models.modules.attention_flash import FlashMultiheadAttention
from PoET.poet.models.modules.packed_sequence import PackedTensorSequences
from ddsm import *
from PoET.poet.models.poet import PoET
from PoET.poet.models.modules.transformer_rotary import RotaryFlashMultiheadAttention
import math
from PoET.poet.models.modules.packed_sequence import (
    PackedTensorSequences,
    get_mask,
    pad_input,
    unpad_input,
)
class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

T = TypeVar("T", Tensor, PackedTensorSequences)

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization, conditioning on timestep embedding
    """
    def __init__(self, dim, time_embed_dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.scale_shift = nn.Linear(time_embed_dim, dim * 2)
        # Initialize to identity
        self.scale_shift.weight.data.zero_()
        self.scale_shift.bias.data.zero_()
        
    def forward(self, x: Tensor, timestep_embedding: Tensor, seqs_cu_seqlens: Tensor) -> Tensor:
        """
        Applies Adaptive Layer Normalization using cumulative sequence lengths.

        Args:
            x: Input tensor (packed sequence). Shape: [packed_len, dim].
            timestep_embedding: Timestep embedding for the batch.
                                Shape: [B, time_embed_dim], where B is the batch size.
            seqs_cu_seqlens: Cumulative sequence lengths for the batch.
                             Shape: [B+1]. Should start with 0 and end with packed_len.

        Returns:
            Normalized and scaled/shifted tensor. Shape: [packed_len, dim].
        """

        # scale_shift_params has shape [B, dim * 2]
        scale_shift_params = self.scale_shift(timestep_embedding)
        # scale and shift both have shape [B, dim]
        scale, shift = scale_shift_params.chunk(2, dim=1)

        # shape [B]
        lengths = seqs_cu_seqlens[1:] - seqs_cu_seqlens[:-1] # Equivalent to seqs_cu_seqlens.diff()

        # Use repeat_interleave to repeat each batch item's scale/shift 'length' times
        # scale_gathered and shift_gathered will have shape [packed_len, dim]
        scale_gathered = scale.repeat_interleave(lengths, dim=0)
        shift_gathered = shift.repeat_interleave(lengths, dim=0)

        # x_norm has shape [packed_len, dim]
        x_norm = self.norm(x)

        # Using the formula: output = x_norm * (1 + scale) + shift
        output = x_norm * (1 + scale_gathered) + shift_gathered

        return output

class DiTTieredTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that operates on sequences-of-sequences. Processes sequences
    in two attention blocks analogously to transformer decoder layers. The first attention
    layer only attends within each sequence. The second attention layer also attends to
    other sequences within each sequence-of-sequences.
    """
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,
        activation=nn.GELU(),
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        time_embed_dim=None,
    ):
        super().__init__()
        
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
            
        # Time embedding dimension
        if time_embed_dim is None:
            time_embed_dim = 4 * d_model
            
        self.dim = d_model
        self.dim_feedforward = dim_feedforward
        self.num_heads = nhead
        self.activation = activation

        self.self_attn = self._init_self_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=False,
        )
        self.multihead_attn = self._init_multi_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=False,
        )

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Zero-initialize the final projection
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

        # Adaptive Layer Norms for timestep conditioning
        self.norm1 = AdaLayerNorm(d_model, time_embed_dim)
        self.norm2 = AdaLayerNorm(d_model, time_embed_dim)
        self.norm3 = AdaLayerNorm(d_model, time_embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence independently.
        """
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )

    def _init_multi_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence-of-sequences.
        """
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.multihead_attn.reset_parameters()
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward_packed(
        self,
        x: PackedTensorSequences,
        timestep_embedding: Tensor,
        seqs_cu_seqlens: Tensor,
        seqs_cu_seqlens_cpu: Optional[Tensor],
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
        return_memory: bool = False,
        patch_size: Optional[int] = None,
        segment_sizes_cpu: Optional[torch.Tensor] = None,
        return_patch: bool = False,
    ) -> Union[
        PackedTensorSequences,
        tuple[PackedTensorSequences, tuple[Tensor, Tensor]],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            tuple[PackedTensorSequences, PackedTensorSequences],
        ],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            tuple[Optional[PackedTensorSequences], Optional[PackedTensorSequences]],
            PackedTensorSequences,
        ],
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: PackedTensorSequences of the individual sequences.
        seqs_cu_seqlens: (B+1,) the cumulative lengths of the sequences-of-sequences.
        src_key_padding_mask: B x N x L x K where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences, and K is the hidden dim
        """
        if return_multi_attention:
            return_attention = True
        if return_self_attention:
            return_attention = True

        # apply the self attention layer on the sequences independently
        x_norm = copy.copy(x)
       
        x_norm.x = self.norm1(x.x, timestep_embedding, seqs_cu_seqlens)
        # at the next line, I get the error forward_padded() takes from 2 to 7 positional arguments but 8 were given
        x2, attn_self = self.self_attn(
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_self_attention,
        )
        x = copy.copy(x)
        x.x = x.x + self.dropout1(x2.x)

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        if patch_size is not None:
            assert patch_size == 0
            assert segment_sizes_cpu is not None
            x_norm.x = x.x[x.cu_seqlens_cpu[:-1].long()]
            x_norm.x = self.norm2(x_norm.x, timestep_embedding, seqs_cu_seqlens)
            n_seqs = (segment_sizes_cpu > 0).long().sum(dim=1)
            x_norm.cu_seqlens = (
                F.pad(n_seqs.cumsum(dim=0), (1, 0))
                .type(torch.int32)
                .to(x_norm.cu_seqlens.device)
            )
            x_norm.cu_seqlens_cpu = F.pad(n_seqs.cumsum(dim=0), (1, 0)).type(
                torch.int32
            )
            x_norm.max_s = n_seqs.max()
            x_norm.positions = None
            nonzero_segment_sizes = (
                segment_sizes_cpu[segment_sizes_cpu > 0].flatten().to(x_norm.x.device)
            )
            assert not x_norm.to_paddedable
            assert src_key_padding_mask is None
        else:
            x_norm.x = self.norm2(x.x, timestep_embedding, seqs_cu_seqlens)
            x_norm.cu_seqlens = seqs_cu_seqlens  # "reshape" the packed sequences
            if seqs_cu_seqlens_cpu is not None:
                x_norm.cu_seqlens_cpu = seqs_cu_seqlens_cpu
            else:
                x_norm.cu_seqlens_cpu = seqs_cu_seqlens.cpu()
            x_norm.max_s = x_norm.cu_seqlens_cpu.max()
            if x_norm.to_paddedable:
                seqs_seqlens = seqs_cu_seqlens.diff()
                x_norm.indices = x_norm.compute_indices(seqs_seqlens)
                x_norm.batch_size = seqs_seqlens.numel()
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.view(
                    -1, src_key_padding_mask.size(-1)
                )
        if not return_memory:
            x2, attn = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
            )
            key, value = None, None
        else:
            x2, attn, (_, key, value) = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
                return_projs=return_memory,
            )
        patch = copy.copy(x2) if return_patch else None
        if patch_size is not None:
            indices = torch.arange(
                nonzero_segment_sizes.numel(), device=x2.x.device
            ).repeat_interleave(nonzero_segment_sizes)
            x2.x = F.embedding(indices, x2.x)
        x = copy.copy(x)
        x.x = x.x + self.dropout2(x2.x)

        x2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(x.x, timestep_embedding, seqs_cu_seqlens)))))
        x.x = x.x + self.dropout3(x2)

        if return_attention:
            return x, (attn_self, attn)
        if return_memory:
            return x, (attn_self, attn), (key, value)
        if return_patch:
            return x, (attn_self, attn), (key, value), patch
        return x

    def forward_padded(
        self,
        x: Tensor,
        timestep_embedding: Tensor,
        seqs_cu_seqlens: Optional[Tensor] = None,
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        Tensor,
        tuple[Tensor, tuple[Tensor, Tensor]],
        tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]], Tensor],
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: Tensor of the individual sequences. Size B x N x L x K
        src_key_padding_mask: B x N x L where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences
        """
        if return_multi_attention:
            return_attention = True
        if return_self_attention:
            return_attention = True

        B, N, L, K = x.size()
        # sequence-independent attention
        x = x.view(B * N, L, K)
        x_norm = self.norm1(x, timestep_embedding)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(B * N, L)
        x2, attn_self = self.self_attn(
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_self_attention,
        )
        x = x + self.dropout1(x2)

        # sequence-of-sequences attention
        x = x.view(B, N * L, K)
        x_norm = self.norm2(x, timestep_embedding)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(B, N * L)
        if not return_memory:
            x2, attn = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
            )
        else:
            x2, attn, (_, key, value) = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
                return_projs=return_memory,
            )
        x = x + self.dropout2(x2)

        # reshape x back
        x = x.view(B, N, L, K)

        x2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(x, timestep_embedding)))))
        x = x + self.dropout3(x2)

        if return_attention:
            return x, (attn_self, attn)
        if return_memory:
            return x, (attn_self, attn), (key, value)
        return x

    def forward(
        self,
        x: T,
        timestep_embedding: Tensor,
        seqs_cu_seqlens: Optional[Tensor] = None,
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        T,
        tuple[T, tuple[Tensor, Tensor]],
        tuple[T, tuple[Optional[Tensor], Optional[Tensor]], T],
    ]:
        """
        See self.forward_padded and self.forward_packed for information about x,
        seqs_cu_seqlens, src_mask, and src_key_padding_mask.

        By default, only returns the output of the layer: (out)

        If return_attention=True, additionally returns the self and multi-sequence
        attention matrices: (out, (attn_self, attn))

        If return_memory=True, additionally returns the "memory" (input to multi-
        sequence attention): (out, (attn_self, attn), memory)
        Here, attn_self and attn may be None depending on the value of
        return_attention.
        """
        # fn = self.forward_padded
        # if type(x) is PackedTensorSequences:
        #     assert seqs_cu_seqlens is not None
        fn = self.forward_packed
        return fn(
            x=x,
            timestep_embedding=timestep_embedding,
            seqs_cu_seqlens=seqs_cu_seqlens,
            seqs_cu_seqlens_cpu=seqs_cu_seqlens_cpu,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_self_attention=return_attention,
            return_multi_attention=return_attention,
            return_memory=return_memory,
        )

class DiTTransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False
    ):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor)
        for layer in self.layers:
            layer.reset_parameters()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(
        self,
        x,
        timestep_embedding, 
        src_mask=None,
        seqs_cu_seqlens: Optional[Tensor] = None,
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_key_padding_mask=None,
        return_attention=False,
        activation_checkpointing=False,
        return_memory = False,
    ):
        attn = []
        for layer in self.layers:
            if not activation_checkpointing:
                x = layer(
                    x=x,
                    timestep_embedding=timestep_embedding,
                    seqs_cu_seqlens=seqs_cu_seqlens,
                    seqs_cu_seqlens_cpu=seqs_cu_seqlens_cpu,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=return_attention,
                    # **kwargs
                    return_memory = return_memory
                )
            else:
                x = checkpoint.checkpoint(
                    layer,
                    x=x,
                    timestep_embedding=timestep_embedding,
                    seqs_cu_seqlens=seqs_cu_seqlens,
                    seqs_cu_seqlens_cpu=seqs_cu_seqlens_cpu,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=return_attention,
                    # **kwargs
                    return_memory = return_memory,
                    use_reentrant=False,
                )
            if return_attention:
                x, a = x
                attn.append(a)

        if return_attention:
            return x, attn

        return x

class DiTTieredRotaryTransformerEncoderLayer(DiTTieredTransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout, 
        batch_first, 
        time_embed_dim,
        rotary_scale=None,
        rotary_force_fp32=None,
        use_multi_rotary=True,
        # **kwargs,
    ):
        
            
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        self.use_multi_rotary = use_multi_rotary
        super().__init__(d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            activation=nn.GELU(),
            dropout=dropout,
            use_qkv_bias=False,
            batch_first=batch_first,
            time_embed_dim=time_embed_dim,
            )

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence independently.
        """
        return RotaryFlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

    def _init_multi_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence-of-sequences.
        """
        Module = FlashMultiheadAttention
        if self.use_multi_rotary:
            Module = RotaryFlashMultiheadAttention
        return Module(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )



class DiT(PoET):
    """A time-dependent score-based model built upon DiT architecture."""

    def __init__(
        self,
        n_vocab: int,
        hidden_dim: int = 768,
        ff_dim: Optional[int] = None,
        num_layers: int = 6,
        nhead: int = 12,
        dropout: float = 0,
        use_multi_rotary: bool = True,
        norm: bool = False,
        mask_token: int = 21,  # kept just to maintain compatability with old models
        time_embed_dim: int = 256
    ):
        super(DiT, self).__init__(
            n_vocab,
            hidden_dim,
            ff_dim,
            num_layers,
            nhead,
            dropout,
            use_multi_rotary,
            norm,
            mask_token,  
        )

        self.decoder = DiTTransformerEncoder(
            encoder_layer=DiTTieredRotaryTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                use_multi_rotary=use_multi_rotary,
                batch_first=True,
                time_embed_dim=time_embed_dim
            ),
            num_layers=num_layers,
        )
        self.time_embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_embed_dim),
                                   nn.Linear(time_embed_dim, time_embed_dim), GELU())
        if norm:
            self.norm = AdaLayerNorm(hidden_dim, time_embed_dim)

        self.project_pt = nn.Linear(n_vocab, hidden_dim)
    
    def forward(self, xs: torch.Tensor, segment_sizes: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute the next token probability distributions.

        Examples:
          Example input with batch size 1

          xs: [$ A B * $ A B C * $ E F]
          segment_sizes: [[4, 5, 3]]
          time:  [[1]] 

          Note that the last sequence in a sequence of sequences does not need to have a
          stop token.

        Args:
          xs:
            (B, L) sequence of sequences of tokens
          segment_sizes:
            (B, N) the lengths of each sequence in the sequence of sequences
          time:
            (B, 1) timestep of each sample in the batch 
        

        Returns:
          (B, L, V) logits of the next token probability distributions. Here, V is
          the vocabulary size

        """
        B, L, _  = xs.size()
        time_embed = self.time_embed(time) # Shape [B, time_embed_dim]
        
        seqs_seqlens = segment_sizes.sum(dim=1).type(torch.int32)
        # print(xs, seqs_seqlens)
        xs, indices, _, _ = unpad_input(xs, ~get_mask(seqs_seqlens))
        h = self.project_pt(xs)
        
        segment_sizes_cpu = segment_sizes.cpu()
        seqs_seqlens_cpu = segment_sizes_cpu.sum(dim=1).type(torch.int32)
    

        nonzero_segment_sizes_cpu = (
            segment_sizes_cpu[segment_sizes_cpu > 0].flatten().type(torch.int32)
        )
        cu_seqlens_cpu = F.pad(
            nonzero_segment_sizes_cpu.cumsum(
                dim=0, dtype=nonzero_segment_sizes_cpu.dtype
            ),
            (1, 0),
        )
        cu_seqlens = cu_seqlens_cpu.to(xs.device)
        h = PackedTensorSequences(
            packed_tensor=h,
            positions=torch.cat(
                [
                    torch.arange(segment_size, dtype=xs.dtype, device=xs.device)
                    for segment_size in nonzero_segment_sizes_cpu
                ]
            ),
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_s=nonzero_segment_sizes_cpu.max(),
            # only needed for unpadding (used in standard attn)
            to_paddedable=False,
            indices=None,
            batch_size=None,
        )
        h = self.decoder.forward(
            h,
            seqs_cu_seqlens=F.pad(
                seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
            ),
            seqs_cu_seqlens_cpu=F.pad(
                seqs_seqlens_cpu.cumsum(dim=0, dtype=seqs_seqlens_cpu.dtype),
                (1, 0),
            ),
            timestep_embedding = time_embed
        )

        logits = self.linear.forward(self.norm(h.x))
        logits, _ = pad_input(logits, indices, B, L)  # (B,L,num_tokens)
        return logits



    
    


