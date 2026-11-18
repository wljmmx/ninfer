"""Exercise the production Python writer's batched bank through the C++ reader and materializer."""

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch
except ImportError:  # The CTest hook follows the selected interpreter's dependencies.
    print("torch is unavailable: select the environment with the conversion dependencies")
    raise SystemExit(77)

from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter

BANK = "batched_block_scale_k16_m128x4_v1"
EXPERTS, ROWS, COLUMNS = 2, 128, 128


def codes() -> torch.Tensor:
    """Expert e's first row begins with the E2M1 word 2 (1.0) in both nibbles."""

    plane = torch.zeros((ROWS, COLUMNS // 2), dtype=torch.uint8)
    plane[0, 0] = 0x22
    return plane


def scales() -> torch.Tensor:
    """Expert e's first K group holds E4M3FN 0x40 (2.0)."""

    plane = torch.zeros((ROWS, COLUMNS // 16), dtype=torch.uint8)
    plane[0, 0] = 0x40
    return plane


def main() -> int:
    executable = sys.argv[1]
    path = Path(tempfile.mkdtemp(prefix="ninfer-writer-bank-")) / "bank.ninfer"
    with ArtifactWriter(
        path,
        [TensorSpec("bank", (EXPERTS, ROWS, COLUMNS), "nvfp4", BANK)],
        components={"text": {"config": {}}},
        bindings={"bank": {"object": "bank"}},
    ) as writer:
        output = TensorOutput(writer, "bank")
        for expert in range(EXPERTS):
            output.write_codes(expert * ROWS, codes(), scales())
        output.write_divisors([torch.tensor(2.0), torch.tensor(3.0)])
    return subprocess.run([executable, "--python-bank", str(path)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
