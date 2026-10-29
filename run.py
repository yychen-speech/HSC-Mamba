import argparse


DATASET_PROFILES = {
    "wsj0-2mix": "clean",
    "libri2mix": "libri2mix",
    "wham": "noisy",
    "whamr": "noisy_reverb",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate HSC-Mamba")
    parser.add_argument(
        "--dataset",
        choices=DATASET_PROFILES,
        default="whamr",
        help="Dataset used for this run",
    )
    parser.add_argument(
        "--engine-mode",
        choices=["train", "test", "test_save"],
        default="train",
        help="Run mode",
    )
    parser.add_argument(
        "--out-wav-dir",
        default=None,
        help="Optional output directory for separated audio in test_save mode",
    )
    args = parser.parse_args()
    args.dm_data = DATASET_PROFILES[args.dataset]
    return args


if __name__ == "__main__":
    from hsc_mamba.main import main

    main(parse_args())
