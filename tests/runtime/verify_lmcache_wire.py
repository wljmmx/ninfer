"""Decode what the C++ encoders emitted, using the classes the server itself uses.

The point is not that the bytes parse -- it is that they parse into the *server's* types with the
values intact. A field renamed or mis-sized would show up here as a validation error rather than as
a mysterious failure at RPC time.
"""

import struct
import sys
from pathlib import Path

import msgspec

from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
failures = 0


def expect(condition, what):
    global failures
    print(f"{'ok  ' if condition else 'FAIL'} {what}")
    if not condition:
        failures += 1


key_bytes = (out_dir / "wire_key.bin").read_bytes()
key = msgspec.msgpack.decode(key_bytes, type=IPCCacheServerKey)
expect(key.model_name == "qwen3.6-35b-a3b", "key.model_name survives")
expect(key.world_size == 1, "key.world_size survives")
expect(key.worker_id == 0, "key.worker_id survives (not null)")
expect(key.start == 0 and key.end == 512, "key token range is 0..512")
expect(key.request_id == "req-wire-0001", "key.request_id survives")
expect(key.cache_salt == "", "key.cache_salt defaults to empty")
expect(key.num_kv_readers == 1, "key.num_kv_readers survives")

tokens = list(struct.unpack(f">{len(key.token_bytes) // 4}I", key.token_bytes))
expect(len(tokens) == 512, f"token_bytes carries 512 tokens (got {len(tokens)})")
expect(all(t == i * 7 + 3 for i, t in enumerate(tokens[:64])),
       "token ids decode back in order, big-endian packed")
# The server derives chunk hashes from exactly these bytes, so a packing error is a cache miss.
expect(len(key.token_bytes) % 4 == 0, "token_bytes is a whole number of 4-byte words")

reg_bytes = (out_dir / "wire_register.bin").read_bytes()
reg = msgspec.msgpack.decode(reg_bytes, type=RegisterEngineDrivenContextPayload)
expect(reg.instance_id == 4242, "register.instance_id survives")
expect(reg.block_size == 256, "register.block_size survives")
expect(reg.num_layers == 40 and reg.hidden_dim_size == 32, "register geometry survives")
expect(reg.dtype_str == "bfloat16", "register dtype_str survives")
expect(reg.use_mla is False, "register.use_mla is a real bool, not an int")
expect(reg.num_physical_slots == 256, "register.num_physical_slots survives")
expect(len(reg.group_layouts) == 2, f"two groups survive (got {len(reg.group_layouts)})")
if len(reg.group_layouts) == 2:
    kv, state = reg.group_layouts
    expect(kv.kv_size == 2 and kv.recurrent is False, "group 0 is attention KV (2 planes)")
    expect(state.kv_size == 1 and state.recurrent is True, "group 1 is the recurrent state group")
    expect(state.dtype_str == "float32" and state.tokens_per_block == 256,
           "group 1 carries its own dtype and block size")

print()
print(f"{failures} failure(s)")
sys.exit(0 if failures == 0 else 1)
