import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dropout import DropoutColumnwise, DropoutRowwise
from einops import rearrange
from torch import einsum
from torch.nn.parameter import Parameter


def init_weights(m):
    # print(m)
    """
    Initialize weights for linear layers.

    This function is intended to initialize the weights and biases of layers that
    are instances of nn.Linear using Xavier uniform initialization for weights
    and setting biases to 0.01. Currently, the initialization code is commented
    out, so the function serves as a placeholder.

    Args:
        m: A neural network module. If m is an instance of nn.Linear, it is a target
           for weight initialization.
    """
    if m is not None and isinstance(m, nn.Linear):
        pass
        # torch.nn.init.xavier_uniform_(m.weight)
        # #torch.nn.init.xavier_normal(m.bias)
        # try:
        #     m.bias.data.fill_(0.01)
        # except:
        #     pass


# mish activation
class Mish(nn.Module):
    def __init__(self):
        """
        Initializes the instance.

        Delegates initialization to the superclass constructor.
        """
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        """
        Computes the Mish activation function.

        This function applies the Mish activation element-wise, defined as:
          x * tanh(softplus(x)).

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor with the Mish activation applied.
        """
        return x * (torch.tanh(F.softplus(x)))


def gem(x, p=3, eps=1e-6):
    """
    Computes the generalized mean (GeM) of the input tensor along its last dimension.

    This function raises each element of the input tensor to the power p, averages the results
    over the last dimension using 1D average pooling, and then takes the p-th root of the averaged result.
    A small epsilon is used to clamp the input tensor for numerical stability.

    Args:
        x (Tensor): Input tensor on which to perform pooling.
        p (float, optional): The exponent parameter controlling the pooling behavior. Defaults to 3.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        Tensor: The pooled tensor with the same number of dimensions as the input, except the last
        dimension is reduced to 1.
    """
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        """
        Initializes the GeM pooling module.

        This module sets up a learnable exponent parameter for the generalized mean pooling
        operation and a small constant for numerical stability.

        Args:
            p (float, optional): Initial exponent value for pooling. Default is 3.
            eps (float, optional): Small constant to avoid division by zero. Default is 1e-6.
        """
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Applies generalized mean pooling to the input tensor.

        This method computes the generalized mean of the input tensor using the module’s
        pooling exponent (p) and stabilization constant (eps), and returns the pooled tensor.
        """
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        """
        Return the string representation of the object.

        The representation includes the class name along with the 'p' parameter (formatted to four decimal places) and the 'eps' value.
        """
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        """
        Initialize the module with temperature scaling and dropout for attention.

        Parameters:
            temperature: Scaling factor applied to attention logits.
            attn_dropout: Dropout probability for attention weights (default is 0.1).
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.gamma=torch.tensor(32.0)

    def forward(self, q, k, v, mask=None, attn_mask=None):

        # print(self.gamma)
        """
        Computes scaled dot-product attention.

        Calculates attention scores by taking the dot product of the query and key
        tensors (with key transposed) and scaling them by a temperature factor. An
        optional bias (mask) and an optional attention mask can be applied to adjust
        these scores before they are normalized with softmax and processed with dropout.
        Finally, the function computes the weighted sum of the value tensor using the
        normalized attention weights.

        Args:
            q: Query tensor.
            k: Key tensor.
            v: Value tensor.
            mask: Optional tensor added as a bias to the attention scores.
            attn_mask: Optional tensor used to mask out specific positions in the
                       attention scores.

        Returns:
            A tuple (output, attn) where output is the result of the attention operation
            and attn contains the normalized attention weights.
        """
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        # to_plot=attn[0,0].detach().cpu().numpy()
        # plt.imshow(to_plot)
        # plt.show()
        # exit()

        # exit()
        if mask is not None:

            attn = attn + mask  # this is actually the bias

        if attn_mask is not None:
            attn = attn.float().masked_fill(attn_mask == -1, float("-1e-9"))

        attn = self.dropout(F.softmax(attn, dim=-1))
        # print(attn[0,0])
        # to_plot=attn[0,0].detach().cpu().numpy()
        # with open('mat.txt','w+') as f:
        #     for vector in to_plot:
        #         for num in vector:
        #             f.write('{:04.3f} '.format(num))
        #         f.write('\n')
        # plt.imshow(to_plot)
        # plt.show()
        # exit()
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        """
        Initializes the multi-head attention module.

        This constructor creates linear projections for queries, keys, and values from the input
        features, and sets up the scaled dot-product attention mechanism with temperature scaling
        based on the query/key dimension. It also initializes a final linear layer to combine the
        attention outputs, along with dropout and layer normalization for improved training stability.

        Args:
            d_model: Dimensionality of the input feature space.
            n_head: Number of attention heads.
            d_k: Dimensionality for the query and key projections per head.
            d_v: Dimensionality for the value projections per head.
            dropout: Dropout probability applied to the attention output.
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, src_mask=None):
        """
        Applies multi-head attention to input queries, keys, and values.

        Projects inputs into multiple attention heads and computes scaled dot-product attention. If a source
        mask is provided, it is transformed into an attention mask via an outer product. The resulting attended
        features are concatenated, processed with dropout and a linear transformation, then combined with a
        residual connection and normalized.

        Args:
            q: Query tensor with shape (batch_size, query_length, feature_dim).
            k: Key tensor with shape (batch_size, key_length, feature_dim).
            v: Value tensor with shape (batch_size, value_length, feature_dim).
            mask: Optional mask tensor to limit attention to specific positions.
            src_mask: Optional binary mask used to generate an additional attention mask via outer product.

        Returns:
            A tuple (output, attn) where output is the normalized tensor of attended queries and attn contains the
            computed attention weights.
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        if src_mask is not None:
            src_mask[src_mask == 0] = -1
            src_mask = src_mask.unsqueeze(-1).float()
            attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1)).unsqueeze(1)
            q, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)
        # print(attn.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # print(q.shape)
        # exit()
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ConvTransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        pairwise_dimension,
        use_triangular_attention,
        dropout=0.1,
        k=3,
    ):
        """
        Initialize a ConvTransformerEncoderLayer module.

        This module integrates multi-head self-attention, a feedforward network, convolutional filtering,
        and pairwise feature transformations to process input sequences. When enabled, it also applies
        triangular attention mechanisms to refine pairwise interactions. The layer is equipped with normalization
        and dropout layers to ensure stable training.

        Args:
            d_model: Dimensionality of the input embeddings.
            nhead: Number of self-attention heads.
            dim_feedforward: Size of the intermediate layer in the feedforward network.
            pairwise_dimension: Dimensionality of the pairwise feature representation.
            use_triangular_attention: Flag indicating whether to apply triangular attention to pairwise features.
            dropout: Dropout probability applied across various submodules (default is 0.1).
            k: Kernel size for the convolutional operation (default is 3).
        """
        super(ConvTransformerEncoderLayer, self).__init__()

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(
            d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        self.conv = nn.Conv1d(d_model, d_model, k, padding=k // 2)

        self.triangle_update_out = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="outgoing"
        )
        self.triangle_update_in = TriangleMultiplicativeModule(
            dim=pairwise_dimension, mix="ingoing"
        )

        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutRowwise(dropout)

        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="row"
            )
            self.triangle_attention_in = TriangleAttention(
                in_dim=pairwise_dimension, dim=pairwise_dimension // 4, wise="col"
            )

            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=d_model, pairwise_dim=pairwise_dimension
        )

        # self.deconv=nn.ConvTranspose1d(d_model,d_model,k)
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension),
        )

    def forward(
        self, src, pairwise_features, src_mask=None, return_attention_weights=False
    ):
        """
        Performs a forward pass that updates sequence and pairwise features via convolution, self-attention, and triangular operations.

        Args:
            src: Input tensor of sequence features, modulated by src_mask.
            pairwise_features: Tensor of pairwise features that is updated through outer product projections and triangle modules.
            src_mask: Optional mask tensor applied to src to suppress invalid entries.
            return_attention_weights: If True, includes self attention weights in the returned tuple.

        Returns:
            A tuple containing the updated sequence features and pairwise features. If return_aw is True, the tuple also includes the self attention weights.
        """
        src = src * src_mask.float().unsqueeze(-1)

        # res = src
        # print(self.norm3(self.conv(src.permute(0,2,1)).permute(0,2,1)).shape)

        src = src + self.conv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.norm3(src)

        pairwise_bias = self.pairwise2heads(
            self.pairwise_norm(pairwise_features)
        ).permute(0, 3, 1, 2)

        src2, attention_weights = self.self_attn(
            src, src, src, mask=pairwise_bias, src_mask=src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        pairwise_features = pairwise_features + self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pair_dropout_out(
            self.triangle_update_out(pairwise_features, src_mask)
        )
        pairwise_features = pairwise_features + self.pair_dropout_in(
            self.triangle_update_in(pairwise_features, src_mask)
        )
        if self.use_triangular_attention:
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(
                self.triangle_attention_out(pairwise_features, src_mask)
            )
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(
                self.triangle_attention_in(pairwise_features, src_mask)
            )
        pairwise_features = pairwise_features + self.pair_transition(pairwise_features)

        if return_attention_weights:
            return src, pairwise_features, attention_weights
        else:
            return src, pairwise_features


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        """
        Initialize the positional encoding module using sine and cosine functions.

        Precomputes positional encodings up to a specified maximum sequence length and stores
        them as a non-trainable buffer. These encodings, generated from sine and cosine functions,
        can be added to input embeddings to inject positional information. A dropout layer is
        applied to the output as a form of regularization.

        Args:
            d_model: Dimensionality of the embeddings.
            dropout: Dropout probability applied to the positional encoding output.
            max_len: Maximum sequence length for which positional encodings are precomputed.
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encodings to the input tensor and apply dropout.

        This method augments the input tensor with precomputed positional encodings,
        sliced to match the input sequence length, and then applies dropout for
        regularization.

        Args:
            x (Tensor): Input tensor with shape [seq_len, batch_size, embedding_dim].

        Returns:
            Tensor: The result after adding positional encodings to x and applying dropout.
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Outer_Product_Mean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        """
        Initializes an Outer_Product_Mean module.

        This module sets up two linear projection layers. The first layer projects input features from in_dim to dim_msa,
        preparing them for an outer product operation. The second layer maps the flattened outer product (of dimension dim_msa²)
        to a pairwise representation of dimension pairwise_dim.

        Args:
            in_dim (int): Dimensionality of the input features (default: 256).
            dim_msa (int): Reduced feature dimension for the MSA projection (default: 32).
            pairwise_dim (int): Output dimensionality of the pairwise feature representation (default: 64).
        """
        super(Outer_Product_Mean, self).__init__()

        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(self, seq_rep, pair_rep=None):
        """
        Computes a pairwise feature representation from sequence features.

        This method applies an initial projection to the input sequence features, computes their outer
        product to capture pairwise interactions, and then rearranges and projects the result into a
        pairwise feature map. If an optional pairwise representation is provided, it is added to the
        computed features.

        Args:
            seq_rep: Tensor containing the sequence features.
            pair_rep: Optional tensor representing an existing pairwise feature map.

        Returns:
            Tensor representing the combined pairwise features.
        """
        seq_rep = self.proj_down1(seq_rep)
        outer_product = torch.einsum("bid,bjc -> bijcd", seq_rep, seq_rep)
        outer_product = rearrange(outer_product, "b i j c d -> b i j (c d)")
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product = outer_product + pair_rep

        return outer_product


class relpos(nn.Module):

    def __init__(self, dim=64):
        """
        Initialize the relative positional encoding module.

        This constructor sets up a linear transformation that projects a fixed input size of 17
        to the specified output dimensionality.

        Args:
            dim (int): The output feature dimension for the linear projection (default: 64).
        """
        super(relpos, self).__init__()

        self.linear = nn.Linear(17, dim)
        self.bin_values = torch.arange(-8, 9)
        self.bdy = torch.tensor(8)

    def forward(self, src):
        """
        Compute relative positional embeddings for an input sequence.

        This method determines the pairwise differences between sequence positions,
        clamps these differences to the range [-8, 8], and converts them into one-hot
        encodings using bin values from -8 to 8. The resulting one-hot tensor is then
        projected via a linear transformation to produce the final relative positional
        embeddings.

        Args:
            src (torch.Tensor): Input tensor whose second dimension determines the sequence length.

        Returns:
            torch.Tensor: The projected relative positional embeddings.
        """
        L = src.shape[1]
        device = src.device

        res_id = torch.arange(L).to(device).unsqueeze(0)
        d = res_id[:, :, None] - res_id[:, None, :]
        d = torch.minimum(torch.maximum(-self.bdy, d), self.bdy)
        d_onehot = (d[..., None] == self.bin_values).float()

        # print(d_onehot.sum(dim=-1).min())
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p


def exists(val):
    """
    Check if the given value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if val is not None, False otherwise.
    """
    return val is not None


def default(val, d):
    """
    Return candidate value if it exists, otherwise return the default.

    This function checks whether the given value meets the existence criteria
    (using the external exists() function). If the value exists, it is returned;
    otherwise, the default value is returned.

    Args:
        val: The candidate value to be tested.
        d: The default value to return if val does not exist.

    Returns:
        The candidate value if it exists; otherwise, the default value.
    """
    return val if exists(val) else d


class TriangleMultiplicativeModule(nn.Module):
    def __init__(self, *, dim, hidden_dim=None, mix="ingoing"):
        """
        Initialize the triangle multiplicative module with gated interactions.

        This constructor sets up projection layers for left and right inputs and
        configures gating mechanisms with identity initialization. It also applies
        layer normalization to both the input features and the intermediate
        representation. An einsum equation is selected based on the specified mixing
        direction to determine how features are multiplicatively combined.

        Args:
            dim: The dimensionality of the input features.
            hidden_dim: Optional; the dimensionality of the hidden representation,
                defaults to the value of dim.
            mix: A string indicating the type of feature mixing, which must be either
                "ingoing" or "outgoing".
        """
        super().__init__()
        assert mix in {"ingoing", "outgoing"}, "mix must be either ingoing or outgoing"

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)

        if mix == "outgoing":
            self.mix_einsum_eq = "... i k d, ... j k d -> ... i j d"
        elif mix == "ingoing":
            self.mix_einsum_eq = "... k j d, ... k i d -> ... i j d"

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, src_mask=None):
        """
        Applies gated multiplicative mixing to a symmetrical feature map.

        Normalizes the input tensor and computes two projected streams that are modulated by an
        optional spatial mask and learnable gating functions. The method then fuses the gated
        projections via a predefined Einstein summation, applies further normalization and
        gating, and finally projects the result to produce the transformed feature map.

        Args:
            x (Tensor): A symmetrical feature map tensor (where the second and third dimensions are equal).
            src_mask (Tensor, optional): A mask tensor that is expanded into a pairwise mask to modulate
                the feature projections.

        Returns:
            Tensor: The output tensor after gated multiplicative mixing and transformation.
        """
        src_mask = src_mask.unsqueeze(-1).float()
        mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))
        assert x.shape[1] == x.shape[2], "feature map must be symmetrical"
        if exists(mask):
            mask = rearrange(mask, "b i j -> b i j ()")

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class RibonanzaNet(nn.Module):

    # def __init__(self, ntoken=5, nclass=1, ninp=512, nhead=8, nlayers=9, kmers=9, dropout=0):
    def __init__(self, config):
        """Initializes RibonanzaNet with transformer, embedding, and decoder modules.

        Constructs the model architecture by assembling a series of ConvTransformerEncoderLayer modules,
        an embedding layer for token inputs, and a linear decoder for classification. It also initializes
        modules for outer product pooling and relative positional encoding to handle pairwise features.
        All relevant hyperparameters (e.g., ninp, nlayers, nhead, k, ntoken, nclass, dropout, pairwise_dimension,
        and use_triangular_attention) are specified via the configuration object.

        Args:
            config: A configuration object containing model hyperparameters.
        """
        super(RibonanzaNet, self).__init__()
        self.config = config

        # input layers
        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.outer_product_mean = Outer_Product_Mean(
            in_dim=config.ninp, pairwise_dim=config.pairwise_dimension
        )
        self.pos_encoder = relpos(config.pairwise_dimension)

        # blocks
        print(f"constructing {config.nlayers} ConvTransformerEncoderLayers")
        self.transformer_encoder = []
        dim_multiplier = 4
        for i in range(config.nlayers):
            if i != config.nlayers - 1:
                k = config.k
            else:
                k = 1
            # print(k)
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp,
                    nhead=config.nhead,
                    dim_feedforward=config.ninp * dim_multiplier,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dropout=config.dropout,
                    k=k,
                )
            )
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        # output layers
        self.decoder = nn.Linear(config.ninp, config.nclass)
        # if config.use_bpp:
        #     self.mask_dense=nn.Conv2d(2,config.nhead//4,1)
        # else:
        #     self.mask_dense=nn.Conv2d(1,config.nhead//4,1)

        self.outer_product_mean = Outer_Product_Mean(
            in_dim=config.ninp, pairwise_dim=config.pairwise_dimension
        )
        self.pos_encoder = relpos(config.pairwise_dimension)

    def forward(self, src, src_mask=None, return_attention_weights=False):
        """
        Performs a forward pass through the model.

        Encodes the input sequence, computes pairwise features using an outer product mean
        and positional encoding, and passes the result through multiple transformer encoder
        layers and a decoder. If requested, collects and returns attention weights from each
        encoder layer.

        Args:
            src: Input tensor of shape (batch_size, sequence_length).
            src_mask: Optional mask tensor for the encoder layers.
            return_attention_weights: If True, returns a tuple containing the output and a list of attention weights.

        Returns:
            If return_aw is True, a tuple (output, attention_weights); otherwise, the output tensor.
        """
        B, L = src.shape
        src = self.encoder(src).reshape(B, L, -1)

        # spawn outer product
        # outer_product = torch.einsum('bid,bjc -> bijcd', src, src)
        # outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        pairwise_features = self.outer_product_mean(src) + self.pos_encoder(src)

        attention_weights = []
        for i, layer in enumerate(self.transformer_encoder):
            if src_mask is not None:
                # src_key_padding_mask
                if return_attention_weights:
                    src, aw = layer(
                        src,
                        pairwise_features,
                        src_mask,
                        return_attention_weights=return_attention_weights,
                    )
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(
                        src,
                        pairwise_features,
                        src_mask,
                        return_attention_weights=return_attention_weights,
                    )
            else:
                if return_attention_weights:
                    src, aw = layer(
                        src,
                        pairwise_features,
                        return_attention_weights=return_attention_weights,
                    )
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(
                        src,
                        pairwise_features,
                        return_attention_weights=return_attention_weights,
                    )
            # print(src.shape)
        output = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0

        if return_attention_weights:
            return output, attention_weights
        else:
            return output


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise="row"):
        """
        Initializes the TriangleAttention module.

        This module prepares the internal layers for computing triangular attention using
        multiple heads and a gating mechanism. It applies layer normalization, linear
        projections to compute query, key, and value tensors, and combines head outputs
        through a final linear transformation. The 'wise' parameter controls whether
        attention is computed row- or column-wise.

        Args:
            in_dim (int, optional): Dimension of the input features. Default is 128.
            dim (int, optional): Dimension for QKV projections per head. Default is 32.
            n_heads (int, optional): Number of attention heads. Default is 4.
            wise (str, optional): Attention mode ('row' or 'column'). Default is "row".
        """
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.to_out = nn.Linear(n_heads * dim, in_dim)
        # self.to_out.weight.data.fill_(0.)
        # self.to_out.bias.data.fill_(0.)

    def forward(self, z, src_mask):
        """
        Computes triangular attention with row or column-wise masking.

        Transforms the input source mask into a pairwise mask for suppressing invalid positions
        and applies multi-head scaled dot-product attention on normalized features. The method
        projects the input into query, key, and value tensors and, based on the attention mode
        specified by the instance attribute 'wise' ('row' or 'col'), reshapes the tensors accordingly.
        It then computes scaled attention scores with an added bias, masks the scores, and applies
        softmax normalization. Finally, a gating mechanism and linear transformation yield the output.

        Args:
            z: Tensor containing input feature representations.
            src_mask: Binary tensor mask indicating valid positions for computing attention.

        Returns:
            Tensor with the computed triangular attention applied.

        Raises:
            ValueError: If the 'wise' attribute is not set to either 'row' or 'col'.
        """

        # spwan pair mask
        src_mask[src_mask == 0] = -1
        src_mask = src_mask.unsqueeze(-1).float()
        attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))

        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(
            lambda x: rearrange(x, "b i j (h d)->b i j h d", h=self.n_heads), (q, k, v)
        )
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** 0.5
        if wise == "row":
            eq_attn = "brihd,brjhd->brijh"
            eq_multi = "brijh,brjhd->brihd"
            b = rearrange(b, "b i j (r h)->b r i j h", r=1)
            softmax_dim = 3
            attn_mask = rearrange(attn_mask, "b i j->b 1 i j 1")
        elif wise == "col":
            eq_attn = "bilhd,bjlhd->bijlh"
            eq_multi = "bijlh,bjlhd->bilhd"
            b = rearrange(b, "b i j (l h)->b i j l h", l=1)
            softmax_dim = 2
            attn_mask = rearrange(attn_mask, "b i j->b i j 1 1")
        else:
            raise ValueError("wise should be col or row!")
        logits = torch.einsum(eq_attn, q, k) / scale + b
        # plt.imshow(attn_mask[0,0,:,:,0])
        # plt.show()
        # exit()
        logits = logits.masked_fill(attn_mask == -1, float("-1e-9"))
        attn = logits.softmax(softmax_dim)
        # print(attn.shape)
        # print(v.shape)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, "b i j h d-> b i j (h d)")
        z_ = self.to_out(out)
        return z_


if __name__ == "__main__":
    from functions import *

    config = load_config_from_yaml("configs/pairwise.yaml")
    model = RibonanzaNet(config).cuda()
    x = torch.ones(4, 128).long().cuda()
    mask = torch.ones(4, 128).long().cuda()
    mask[:, 120:] = 0
    print(model(x, src_mask=mask).shape)

    # tri_attention=TriangleAttention(wise='row')
    # dummy=torch.ones(6,16,16,128)
    # src_mask=torch.ones(6,16)
    # src_mask[:,12:16]=0
    # out=tri_attention(dummy, src_mask, )
    # print(out.shape)
