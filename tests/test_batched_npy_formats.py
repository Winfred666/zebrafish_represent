import numpy as np

from utils.gen_pretext_dataset import create_masked_sample, generate_pretext_dataset


def test_create_masked_sample_accepts_channel_first_4d():
    vol = np.random.rand(1, 8, 10, 12).astype(np.float32)
    sample = create_masked_sample(vol, mask_type="block", mask_ratio=0.3)
    assert set(sample.keys()) == {"input", "target", "mask"}
    assert tuple(sample["input"].shape) == (1, 8, 10, 12)
    assert tuple(sample["target"].shape) == (1, 8, 10, 12)
    assert tuple(sample["mask"].shape) == (1, 8, 10, 12)


def test_generate_pretext_dataset_handles_batched_5d(tmp_path):
    # Create a fake batched NPY (N,C,D,H,W)
    arr = np.random.rand(3, 1, 6, 7, 8).astype(np.float32)
    npy_path = tmp_path / "batched.npy"
    np.save(npy_path, arr)

    out_pt = tmp_path / "out.pt"
    generate_pretext_dataset(
        [str(npy_path)],
        str(out_pt),
        samples_per_volume=2,
        mask_type="patch",
        mask_ratio=0.5,
        patch_size=2,
    )

    assert out_pt.exists()
