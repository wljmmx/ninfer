# Per-token logprob comparison

`kl_compare.py` compares two per-token logprob record files and prints only the quantities the record
format can actually support. A column it cannot compute is printed as `n/a`; it never fills one with a
guess.

## Records produced by the engine

`ninfer-perplexity --token-logprobs FILE` writes one row per scored token (`apps/perplexity/main.cpp`):

```text
# index	token	logprob
1	1234	-2.91732822
2	5678	-0.00411273
```

| Field | Type | Meaning |
|---|---|---|
| `index` | integer | token position inside its stream; `plan_windows` targets every token after `x[0]`, so a stream of `N` tokens yields positions `1..N-1` |
| `token` | integer | the observed corpus token id at that position |
| `logprob` | float | `log p(x[index] \| x[0..index))` under the evaluated artifact |

Requirements:

- The `#` header line must be present. Without it the tool refuses to guess the column order.
- Both files must describe the same corpus, context and stride, so the `index` and `token` columns
  agree position by position. A mismatch is an error that names both files and lines; override it with
  `--allow-token-mismatch` only when you know why the tokens differ.
- Fields must be finite, and `logprob` must be `<= 0`.
- A record set may be multi-stream (indexes restart at 1 for each stream); pairing is by record order,
  and the bucket columns then pool equal positions across streams.

The optional JSONL form carries the same three keys plus an optional `topk` candidate list:

```json
{"index": 1, "token": 1234, "logprob": -2.91732822,
 "topk": [{"token": 1234, "logprob": -2.91732822}, {"token": 77, "logprob": -3.5}]}
```

`topk` entries may also be `[token, logprob]` pairs. They must be unique and ordered by decreasing
logprob, because element 0 defines the argmax used for top-1 disagreement. The plain
`--token-logprobs` export above does not carry `topk`; `--token-topk` writes a different file for
that purpose (below), and `kl_div.py` reads it.

## Usage

Score the same corpus with two artifacts (or two KV layouts) and keep the two logs:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype fp8 --token-logprobs out/a.tsv
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --kv-dtype rk4v4 --token-logprobs out/b.tsv
python tools/perplexity/kl_compare.py out/a.tsv out/b.tsv
```

```bash
# custom position boundaries, machine-readable output
python tools/perplexity/kl_compare.py a.jsonl b.jsonl --buckets 1024,2048,4096,8192 --json cmp.json
# JSON on stdout
python tools/perplexity/kl_compare.py a.tsv b.tsv --json
```

`--buckets` is a list of half-open position boundaries. The default `1024,2048,4096` produces
`0-1023`, `1024-2047`, `2048-4095` and `4096+`, plus an `all` row.

Exit status is 0 on success and 1 on any input problem, with `file:line: field` in the message.

## Columns

| Column | Definition |
|---|---|
| `tokens` | aligned positions in the bucket |
| `dlogp` | `mean(log p(x) - log q(x))` in nats per token, equal to `CE_Q - CE_P`; positive means P is less surprised by the corpus |
| `dbits` | `dlogp / ln 2` |
| `PPL_P`, `PPL_Q` | `exp(-mean logprob)` of each side, the same definition `apps/perplexity/evaluation.cpp` uses |
| `P/Q` | `PPL_P / PPL_Q`, equivalently `exp(mean_q - mean_p)` |
| `KL(P\|\|Q)` | only from `topk`; see the limits below |
| `top1diff` | fraction of positions where the two `topk` argmaxes differ |

## Top-K export and KL

`ninfer-perplexity --token-topk K` writes `token_topk.tsv` into its `--output` directory:

```text
# index	token	logprob	log_normalizer	topk_ids	topk_logprobs
1	70665	-3.85873127	13.7337313	220,8455,2724,...	-3.85873127,-4.35873127,...
```

| Field | Type | Meaning |
|---|---|---|
| `index`, `token`, `logprob` | integer, integer, float | exactly the columns of the plain export: the position, the observed corpus token, and its log-probability |
| `log_normalizer` | float | `log(sum_v exp(logit_v))` of the same position, i.e. the normalizer the log-probabilities are taken against |
| `topk_ids` | list of integers | the `K` best tokens at that position, ordered by logit descending with the lower id breaking ties |
| `topk_logprobs` | list of floats | those candidates' log-probabilities; element 0 is the position's argmax |

`K` must be in `[1,64]` (`kScoreTopkCapacity`). The candidate list is a genuine top-K of the
vocabulary column, not a sample, and `apps/perplexity` reports the argmax candidate's logprob
bit-identically to what `--token-logprobs` prints for the same token (`ninfer_logits_topk_test`,
`ninfer_qwen3_5_score_topk_real_test`).

`kl_div.py` compares two such files:

```bash
python tools/perplexity/kl_div.py a/token_topk.tsv b/token_topk.tsv
python tools/perplexity/kl_div.py a/token_topk.tsv b/token_topk.tsv --json kl.json
python tools/perplexity/kl_div.py a/token_topk.tsv b/token_topk.tsv --segments 512,1024,2048
```

The first file is p (the reference), the second is q. Each position contributes

```text
KL = sum over shared candidates i of  p_i * (log p_i - log q_i)
   + P_bucket * (log P_bucket - log Q_bucket)
```

where `P_bucket`/`Q_bucket` are each side's mass outside the candidates the two lists share - the
candidates the other model does not rank plus the residual `1 - coverage` the export does not cover.
That is where q's tail mass is charged, under p's own weight. Because both sides then sum to 1 over
`{shared candidates, bucket}`, the value is a KL between two normalized distributions and cannot be
negative, and two identical runs give exactly `0`. Every segment also reports `coverage_p`,
`coverage_q`, the number of p candidates charged to the bucket and the bucket's mass, because the
bucket aggregates tokens whose individual probabilities were never exported.

Default segments end at index 1K, 2K and 4K (`1K`, `2K`, `4K`, `4K+` plus a total row). The JSON
form carries the same numbers with full precision. `tests/test_kl_div.py` pins the estimator on
hand-computable distributions (identical -> 0, a closed-form pair, one missing candidate, many
missing candidates, unequal candidate counts, and mass outside a fully covered list).

## Known limits

- **The plain TSV cannot produce KL.** `ninfer-perplexity` records the observed token only, so both
  sides give one number per position. `KL(P||Q)` needs `p` and `q` over the same support; the tool
  prints `n/a` instead. Use `--token-topk` and `kl_div.py` when a KL is what is wanted.
- **The plain TSV cannot produce top-1 disagreement.** The recorded token is the corpus token, not
  either model's argmax.
- **`dlogp` is a cross-entropy difference, not a KL.** `mean(log p - log q) = CE_Q - CE_P`. It equals
  `KL(P||Q)` only when the scored tokens were sampled from `P`, which a fixed natural corpus does not
  guarantee. Compare it as a likelihood difference between two artifacts on the same text.
- **`topk` KL is restricted.** The sum runs over the union of the two candidate lists, and the mass
  outside each list is lumped into a single residual bucket, which is exact only if `P` and `Q` split
  their residual identically. If `P` puts mass on a token `Q` gives zero probability, the bucket
  reports `n/a` and the count of such positions instead of an infinite value.
- **Buckets are position buckets, not context buckets.** Each token after `x[0]` is scored once, but a
  stride smaller than the context means later positions were scored with truncated local history, so a
  `4096+` bucket mixes positions whose available context differs.
- The tool reads records; it never runs the engine, so it cannot recover information the evaluator did
  not write. Producing a real KL number requires extending the evaluator's logprob export first.
