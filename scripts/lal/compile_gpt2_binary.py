#!/usr/bin/env python3
"""Generate and compile GPT-2 binary layers one at a time to avoid OOM.

Instead of loading all 48 matrices into memory at once (which OOMs at 4GB),
we compile each layer separately (4 matrices), then concatenate the C files.
"""
import subprocess
import sys
import os

LAL_PY = "/home/z/my-project/scripts/lal/lal.py"
WEIGHTS = "gpt2_weights.bin"
OUT_DIR = "/home/z/my-project/download/lal-mvp/src"

def gen_layer_lal(layer):
    """Generate a .lal file for one layer's 4 matrices."""
    dims = ", ".join(str(i) for i in range(768))
    lines = [
        f"# Layer {layer} binary matrices",
        f"bound all_dims = [{dims}]",
        "",
        f'relate l{layer}_attn_qkv(x) = linear(x, "{WEIGHTS}", "h.{layer}.attn.c_attn", threshold=0.0)',
        f'relate l{layer}_attn_proj(x) = linear(x, "{WEIGHTS}", "h.{layer}.attn.c_proj", threshold=0.0)',
        f'relate l{layer}_mlp_fc(x) = linear(x, "{WEIGHTS}", "h.{layer}.mlp.c_fc", threshold=0.0)',
        f'relate l{layer}_mlp_proj(x) = linear(x, "{WEIGHTS}", "h.{layer}.mlp.c_proj", threshold=0.0)',
        "",
        "rule dummy(query):",
        f"    h = l{layer}_attn_qkv(query)",
        "    output(h)",
    ]
    path = f"/tmp/gpt2_layer_{layer}.lal"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path

def compile_layer(layer):
    """Compile one layer's .lal to C."""
    lal_path = gen_layer_lal(layer)
    c_path = f"{OUT_DIR}/gpt2_binary_layer_{layer}.c"
    print(f"[*] compiling layer {layer}...", flush=True)
    r = subprocess.run(
        ["python3", LAL_PY, lal_path, "dummy", c_path, "--binarize"],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        print(f"[!] layer {layer} failed: {r.stderr[:500]}")
        return None
    print(f"    {r.stdout.strip().split(chr(10))[-1] if r.stdout else 'ok'}")
    return c_path

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_c_paths = []
    for layer in range(12):
        c_path = compile_layer(layer)
        if c_path is None:
            print(f"[!] stopping at layer {layer}")
            break
        all_c_paths.append(c_path)

    # Concatenate all layer C files into one
    combined = f"{OUT_DIR}/gpt2_binary_layers.c"
    with open(combined, "w") as out:
        out.write("/* Combined GPT-2 binary layers — all 48 weight matrices binarized. */\n")
        out.write("#include <stdint.h>\n\n")
        for c_path in all_c_paths:
            with open(c_path) as f:
                content = f.read()
            # Strip the #include lines and main() from each file
            lines = content.split("\n")
            skip = False
            for line in lines:
                if line.startswith("#include"):
                    continue
                if line.startswith("int main("):
                    skip = True
                if skip:
                    if line == "}":
                        skip = False
                    continue
                out.write(line + "\n")
            out.write("\n")
    print(f"\n[*] combined: {combined}")
    print(f"[*] size: {os.path.getsize(combined)/1e6:.1f} MB")

if __name__ == "__main__":
    main()
