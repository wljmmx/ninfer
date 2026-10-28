# LMCache checkpoint cache

The server can keep checkpoints in an LMCache instance and restore them into a
later process request. It is off unless `NINFER_LMCACHE_RESTORE` is set.

Status: **restore path only.** Nothing publishes checkpoints yet, so a restore
can only be exercised against data put there by a test. The three-state
decision (local hit / cache hit / prefill) is in place and a miss is free.

## The two sides must be configured to agree

LMCache addresses cached data in **chunks**, and the chunk size is a server
setting (`--chunk-size`). This engine's chunk is
`NINFER_LMCACHE_CHUNK_TOKENS` tokens, defaulting to two pages (128 tokens).

**The server does not expose its value.** `chunk_size_provider` feeds
coordinator registration only, and the `register` reply carries
`shm_name`, `pool_size` and `null_block_id` -- nothing about chunking. So
there is no way to detect a mismatch at startup, and the failure is silent in
both directions:

| Mismatch | What you see |
|---|---|
| Different chunk size | Every lookup misses. Nothing logs an error. |
| Range end not a chunk multiple | The server raises, sends no reply frame, and the request stalls until the client's receive timeout. |

Startup prints both numbers so they can be compared:

```
[lmcache] attached | page 1310720 B, 64 token(s), chunk 128 token(s) (= 2 page(s))
          -- the server must run --chunk-size 128
```

Changing one side means changing the other.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `NINFER_LMCACHE_RESTORE` | unset (off) | `1` attaches the cache |
| `NINFER_LMCACHE_ENDPOINT` | `tcp://127.0.0.1:6555` | ZMQ endpoint of `lmcache server` |
| `NINFER_LMCACHE_CONNECTIONS` | `4` | Connections, and therefore concurrency: the server's unit of parallelism is the connection, not the instance |
| `NINFER_LMCACHE_CHUNK_TOKENS` | 2 pages (128) | Must equal the server's `--chunk-size` |

## Server side

```
lmcache server --transport zmq --port 6555 --http-port 18080 \
    --supported-transfer-mode engine_driven \
    --chunk-size 128 --max-workers 4 \
    --l1-size-gb 10 --eviction-policy LRU
```

`--max-workers` is what makes `NINFER_LMCACHE_CONNECTIONS` worth having: fewer
workers than connections and the connections serialise onto shared slots;
fewer connections than workers and the workers sit idle.

## What a restore costs

About 1.3 s for a 640-token prefix (77 MB: 5 chunks of KV plus the whole state
image), against roughly 8.5 s to prefill the same prefix. The state image is
64 MB of that and travels as a single tensor addressed to the trailing chunk --
see `src/core/lmcache_bridge.h` for why it is not sharded.

## Known gaps

- No publish path yet, so the cache stays empty.
- Restore is synchronous: the lookup and transfer happen on the requesting
  worker. `core/prefetch.h` is the intended home for making it asynchronous,
  and it is not wired.
- A restore borrows device pages for its duration even though it keeps only the
  Host copy. With a 128-page pool, concurrent restores can run it out, and the
  symptom is a cache that "does not work".
- The fingerprint travels in `cache_salt`. Two runs that differ only in KV
  format share the model name and the tokens, so without it a format change
  reads the old format's bytes without an error.
