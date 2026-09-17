from __future__ import annotations

from pathlib import Path
import py_compile
import shutil
import sys

from .common import HYPERQUANT_REF, MF_REF, MODEL_ID, ROOT, UPSTREAM, output


def ok(msg: str) -> None:
    print(f"[HYPERQUANT INTEGRATION OK] {msg}", flush=True)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"[HYPERQUANT INTEGRATION FAIL] {msg}")
    ok(msg)


def git_head(path: Path) -> str:
    return output(["git", "rev-parse", "HEAD"], cwd=path)


def check_sources() -> None:
    for path in (ROOT / "scripts").glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    ok("all local Python sources compile")

    hq = UPSTREAM / "HyperQuant"
    mf = UPSTREAM / "Model-Fingerprint"
    require(git_head(hq) == HYPERQUANT_REF, "HyperQuant is exactly pinned upstream commit")
    require(git_head(mf) == MF_REF, "Model-Fingerprint is exactly pinned upstream commit")

    q = (hq / "integrations" / "llama" / "int8_linear.py").read_text(encoding="utf-8")
    require('from torch.utils.cpp_extension import load' in q, "HyperQuant uses upstream CUDA extension loader")
    require("convert_linears_to_int8" in q and "lattice_pack" in q, "HyperQuant upstream weight-VQ INT8 path is present")
    require('f"-arch={_CUDA_ARCH}"' in q, "HyperQuant CUDA arch patch is runtime-selected")
    require('extra_cuda_cflags=["-O3", "-arch=sm_90a"]' not in q, "hard-coded SM90a flag removed from active INT8 path")

    decoder_cu = (hq / "cuda" / "stage2_cuda_decoder.cu").read_text(encoding="utf-8")
    decoder_h = (hq / "cuda" / "stage2_cuda_decoder.h").read_text(encoding="utf-8")
    require('cudaMalloc(&d_byte_out_' not in decoder_cu,
            "HyperQuant fused INT8/FP8 first-forward path has no second per-layer byte allocation")
    require('d_out = d_out_;' in decoder_cu,
            "HyperQuant fused INT8/FP8 decode reuses the decoder-owned byte buffer")
    require('const uint8_t* device_byte_output() const { return d_out_; }' in decoder_h,
            "HyperQuant INT8 GEMM consumes the reused decoder byte buffer")
    require('decode_fused_out_kernel<OutputDtype::kInt8>' in decoder_cu,
            "upstream fused E8int decode kernel remains present")

    report = mf / "report_FSR_sft_chat.py"
    create = mf / "create_fingerprint_chat.py"
    require(report.exists() and "calc_FSR_from_jsonl" in report.read_text(encoding="utf-8"),
            "official IF-SFT FSR scorer is available")
    require(create.exists(), "official IF-SFT dataset creator is available")


def check_runtime_and_model_contract() -> None:
    import torch
    from transformers import AutoConfig
    from fastchat.model.model_adapter import get_conversation_template

    require(torch.cuda.is_available(), "CUDA visible to existing system Torch")
    cc = torch.cuda.get_device_capability(0)
    require(cc >= (8, 0), f"GPU compute capability {cc} satisfies HyperQuant SM80+ requirement")
    require(shutil.which("nvcc") is not None, "nvcc is available for HyperQuant JIT extension")
    require(get_conversation_template("vicuna") is not None, "FastChat Vicuna template used by IF-SFT is available")

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    require(getattr(cfg, "model_type", None) == "llama", "IF-SFT checkpoint is a LLaMA-family model")
    require(getattr(cfg, "hidden_size", None) == 4096, "IF-SFT checkpoint hidden_size=4096")
    require(getattr(cfg, "intermediate_size", None) == 11008, "IF-SFT checkpoint intermediate_size=11008")
    require(getattr(cfg, "num_hidden_layers", None) == 32, "IF-SFT checkpoint has 32 transformer layers")
    require(4096 % 256 == 0 and 11008 % 256 == 0,
            "LLaMA-2-7B linear contraction dimensions are compatible with HyperQuant RHT=256")


def check_cuda_extension_and_vq() -> None:
    """Build/load the real upstream CUDA extension and exercise 3/4-bps VQ.

    The forward is run twice for each condition so both first-use and steady-state
    decoder behavior are exercised before the 7B model is touched.
    """
    import torch
    import torch.nn as nn
    from hyperquant import calibrate_lattice_bps_to_snr, lattice_alpha
    from integrations.llama.rice_linear import convert_linears
    from integrations.llama.int8_linear import LatticeLinear, _get_int8_ext

    ext = _get_int8_ext()
    require(ext is not None, "HyperQuant upstream lattice_int8_ext compiled and loaded")

    for bps in (4.0, 3.0):
        snr = calibrate_lattice_bps_to_snr(
            lattices=["e8int"], target_bps_list=[bps]
        )["e8int"][bps]["snr_db"]
        alpha = lattice_alpha(snr, "e8int")

        toy = nn.Sequential(nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)).eval()
        stats = convert_linears(toy, skip=(), alpha=alpha, mma="int8", hadamard=256, verbose=False)
        require(stats["n_converted"] == 1, f"HyperQuant {bps:g}-bps converted a real CUDA bf16 nn.Linear")
        require(isinstance(toy[0], LatticeLinear), f"HyperQuant {bps:g}-bps installed upstream LatticeLinear")
        x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            y1 = toy(x)
            y2 = toy(x)
        require(tuple(y1.shape) == (2, 256) and torch.isfinite(y1).all().item(),
                f"HyperQuant {bps:g}-bps first quantized forward is finite and shape-correct")
        require(tuple(y2.shape) == (2, 256) and torch.isfinite(y2).all().item(),
                f"HyperQuant {bps:g}-bps repeated quantized forward is finite and shape-correct")
        del toy, x, y1, y2
        torch.cuda.empty_cache()


def main() -> None:
    print("[HYPERQUANT] preflight starts; it compiles the CUDA extension but does NOT download/quantize the 7B weights", flush=True)
    check_sources()
    check_runtime_and_model_contract()
    check_cuda_extension_and_vq()
    ok("ALL HYPERQUANT PREFLIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
