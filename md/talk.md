## Demystifying GPT-3
&ldquo; attention is all you need &rdquo;

>>>
### Opening remarks
- This talk explains the model architecture of a powerful _sequence-to-sequence_ model called GPT-3.
- Note that an RL task is also a seq2seq task. But this particular one is not well-suited for RL. But [similar ideas](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#snail) apply to RL.
- Previously, seq2seq tasks were the bread and butter of RNNs

>>>
### RNNs 101
![](img/rnn-charseq.jpeg) <!-- .element height="65%" width="65%" -->

[(source)](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

>>>
### TL;DR
If you don't want to use RNNs, you can:

- Encode the sequentialness deterministically in the input
- Drown your model in (masked) self-attention layers

Why would you want to do this?
- Larger input size
- Better results (in some cases)
>>>
### The original paper
![paper](img/attention-paper.png)

For this talk, I borrow heavily from
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
by Harvard NLP.

VVV

![](img/attention-abstract.png)

>>>
### Today

![](img/arch.png)

>>>
### The input
Inputs to a transformer are sequences of (often) words, e.g. a sentence:

<pre>
Die Protokolldatei kann heimlich per E-mail oder FTP 
an einen bestimmten Empfänger gesendet werden.
</pre>

(in the real model, these are preprocessed a bit into finer grammatical units than words)

VVV

Example:
<pre>
▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP 
▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .
</pre>

>>>
Then, sequences are turned into tensors by using the training data
as a vocabulary:
```python3
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, 
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and 
                                         len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 1
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
```
VVV
```python3
SRC.numericalize([tokenize_de("Habe ach, nun, Philosophie, Juristerei und Medizin, und leider auch Theologie")])

=> tensor([[ 8818],
           [ 9613],
           [    2],
           [  181],
           [    2],
           [ 3255],
           [    2],
           [    0],
           [    5],
           [ 1074],
           [    2],
           [    5],
           [ 1567],
           [   58],
           [22073]])
```
>>>
### Embedding layer

The goal of the next step is to embed the discrete space of words $V$ into $\mathbb{R}^{d_\text{model}}$,
where $d_\text{model} << |V|$:
```python3
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```
Note: this embedding is initialized randomly
>>>
### Positional encoding
The next step is a bit puzzling.

![](img/posenc.png)

>>>
### Recall: The time element in RNNs 
In a RNN, when the (n+1)-th word is fed into the network, the previous n words have already been passed forward through the network.
Hence these words are already a bit 'tarnished' by time. 

Plus, whenever two words $w$, $v$ are $k$ words apart, their 'difference' in forward passes is always the same,
regardless of their absolute position.

>>>
### What is positional encoding?
After the embedding layer of the transformer, the input is now a sequence of words $\vec{w}$, where each `$w_i \in \mathbb{R}^{d_\text{model}}$`.

This sequence is fed all at once to the network:
- There is no 'tarnishing' by time
- Similarly, there is no encoded 'relative position'

Positional encoding remedies this by adding a _deterministic_ version of this 'time element'.

>>>
### Adding sines 
To a vector in position $t$, positional encoding adds the following constant:
`
$$
e(t) = \left(
\begin{matrix}
\sin(\omega_1 t) \\
\cos(\omega_1 t) \\
\vdots \\
\sin(\omega_{d_{\text{model}} / 2} t) \\
\cos(\omega_{d_{\text{model}} / 2} t)
\end{matrix}
\right)
$$`
where `$\omega_k = {10000^{- \frac{2k}{d_\text{model}}}}$`

>>>
### Picture
![](img/posenc_plot.png)

[(source)](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

>>>
### Shifting the encoding is linear
Exercise: for every `$k$`, there is a matrix `$T_k$` such that:
`$$
e(t + k) = T_k e(t)
$$`
for all `$t$`. [Solution](https://timodenk.com/blog/linear-relationships-in-the-transformers-positional-encoding/).

This should allow the model 'to easily learn to attend by relative positions'.

>>>
### The attention layer
![](img/attention-layer.png)

>>>
### Keys, queries and values
The attention head is described using some auxiliary terminology.
For self-attention, an input word `$\vec{w} \in \mathbb{R}^{d_\text{model}}$` is
transformed by a linear network layer into a _key_ and a corresponding _value_:
`$$
 \vec{k} = W^K \vec{w} \qquad \vec{v} = W^V \vec{w} 
 $$`

On top of this, each word may be transformed into a _query_, which has the same dimension as a key.
`$$
\vec{q} = W^Q \vec{w}
$$`

VVV

```python3
query, key, value = \
     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
      for l, x in zip(self.linears, (query, key, value))]
```

>>>
### Scaled dot-product attention
Given any key `$\vec{k}$`, a query `$\vec{q}$` determines how much
attention to pay to its corresponding value `$\vec{v}$`:
`$$
\text{Attention}(\vec{q}, \vec{k}, \vec{v}) = \text{softmax}\left(\frac{\vec{q}\cdot\vec{k}}{\sqrt{d_\text{model}}} \right) \vec{v}
$$`
Where `$\cdot$` takes the dot-product.
>>>
### Implementation
In practice, the attention is computed for all query-key pairs at once and summed.
So there are matrices `$Q$`, `$K$`, `$V$` and
`$$
\text{Attention}({Q}, {K}, {V}) = \text{softmax}\left(\frac{{Q}{K}^T}{\sqrt{d_\text{model}}} \right) {V}
$$`
Hence `$V$` is multiplied by an &lsquo;attention matrix&rsquo; in the attention head.
VVV
```python3
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```
>>>
### Multi-head attention

The _multi-head_ combines the output of several attention heads to
'allow the model to jointly attend to information from different representation subspaces at different positions'.

![](img/attention-multi-head.png)

>>>
### What does this look like?
One can plot the output of attention weights `$\text{softmax}\left(\frac{{Q}{K}^T}{\sqrt{d_\text{model}}} \right)$`:

![](img/attention-plot.png)
VVV
```python3
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()
```
VVV
It is easy for the model to attend by relative position. If it learns
the matrix `$T_k$`, then assuming a sequence of words `$X$` is random,
`$\left((T_k X)X^T\right)_{ij}$` is large when `$i = j + k$` or `$i = j$` and averages
to `$0$` otherwise.
>>>
### The decoder
The decoder uses _masked_ self-attention layers. There, the `$i$`-th output of
a self-attention layer depends only on the first `$i$` inputs.

![](img/attention-dot.png)

>>>
### The forward pass
Finally, the self-attended output is added to the raw output and fed through an
identical fully connected network at each position.
`$$
\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2
$$`
![](img/forward-pass.png)
VVV
```python3
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

>>>
### Putting it all together
![](img/arch.png)
VVV
### Encoder and decoder
- In encoder-decoder attention, the encoder provides the keys and the values.
The decoder 'queries' the keys and has access to all of them, i.e. the entire sequence,
unlike in its self-attention layers.

- The decoder input is shifted right. This is in order to bootstrap the translation.
- Note there are no convolutional layers
