"""Extract perceptual encoder weights from a Lightning checkpoint.

Usage: python utils/script/extract_encoder_ckpt.py <lightning_ckpt> <output_path>
"""
import sys
import torch

def extract(lightning_ckpt_path: str, output_path: str) -> None:
    ckpt = torch.load(lightning_ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"]

    # Strip Lightning "model." prefix
    extracted = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            extracted[k[len("model."):]] = v

    if not extracted:
        print("ERROR: No 'model.'-prefixed keys found in state_dict")
        sys.exit(1)

    torch.save(extracted, output_path)
    print(f"Extracted {len(extracted)} keys from {lightning_ckpt_path} -> {output_path}")

    # Print key categories
    prefixes = set(k.split(".")[0] for k in extracted)
    print(f"Top-level prefixes: {prefixes}")

    # Print total param count
    total_params = sum(v.numel() for v in extracted.values())
    print(f"Total parameters: {total_params:,} ({total_params*4/1024/1024:.1f} MB float32)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <lightning_ckpt> <output_path>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
