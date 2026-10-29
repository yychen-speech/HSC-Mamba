import os
import torch
import random
import librosa as audio_lib
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly

from ..utils import util_dataset
from ..utils.decorators import *
from loguru import logger
from torch.utils.data import Dataset, DataLoader


STATISTICS_DIR = os.path.join(os.path.dirname(__file__), "statistics")
NOISE_SNR_DISTRIBUTIONS = {
    "noisy": os.path.abspath(os.path.join(STATISTICS_DIR, "wham_train_snr_db.csv")),
    "noisy_reverb": os.path.abspath(
        os.path.join(STATISTICS_DIR, "whamr_train_snr_db.csv")
    ),
}


def _load_noise_snr_distribution(path):
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=1, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid noise SNR distribution: {path}")
    return values


def _sample_empirical_noise_snr(values):
    return float(values[random.randrange(values.size)])


def _pad_to_stride(wav, stride):
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    valid_len = int(wav.shape[0])
    pad_len = (-valid_len) % stride
    if pad_len == 0:
        return wav, valid_len
    pad_width = [(0, pad_len)] + [(0, 0)] * (wav.ndim - 1)
    return np.pad(wav, pad_width), valid_len


@logger_wraps()
def get_dataloaders(args, dataset_config, loader_config, seed=2026):
    partitions = ["test"] if "test" in args.engine_mode else ["train", "valid", "test"]
    dataloaders = {}
    data_type = dataset_config.get("data_type", "unknown")

    for partition in partitions:
        partition_config = dataset_config[partition]
        scp_dir = dataset_config["scp_dir"]
        scp_config_mix = os.path.join(scp_dir, partition_config["mixture"])
        spk_keys = [key for key in ("spk1", "spk2") if key in partition_config]
        scp_config_spk = [os.path.join(scp_dir, partition_config[key]) for key in spk_keys]
        reverb_keys = [f"{key}_reverb" for key in spk_keys if f"{key}_reverb" in partition_config]
        scp_config_reverb = [os.path.join(scp_dir, partition_config[key]) for key in reverb_keys]
        scp_config_noise = os.path.join(scp_dir, partition_config["noise"]) if "noise" in partition_config else None
        dynamic_mixing = partition_config.get("dynamic_mixing", False) if partition == "train" else False
        speed_list = partition_config.get("speed_list") if partition == "train" else None

        dataset = MyDataset(
            max_len=dataset_config["max_len"],
            fs=dataset_config["sampling_rate"],
            partition=partition,
            data_type=data_type,
            wave_scp_srcs=scp_config_spk,
            wave_scp_mix=scp_config_mix,
            wave_scp_noise=scp_config_noise,
            wave_scp_reverbs=scp_config_reverb if scp_config_reverb else None,
            dynamic_mixing=dynamic_mixing,
            speed_list=speed_list,
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed) + {"train": 0, "valid": 1, "test": 2}[partition])
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1 if partition == "test" else loader_config["batch_size"],
            shuffle=partition == "train",
            pin_memory=loader_config["pin_memory"],
            num_workers=loader_config["num_workers"],
            drop_last=loader_config["drop_last"],
            collate_fn=_collate,
            generator=generator,
        )
        dataloaders[partition] = dataloader
    return dataloaders


def _collate(egs):
    def __prepare_target_rir(dict_list, index):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(d["src"][index], dtype=torch.float32) for d in dict_list],
            batch_first=True,
        )

    if type(egs) is not list:
        raise ValueError("Unsupported index type({})".format(type(egs)))

    num_spks = len(egs[0]["src"])
    dict_list = sorted([eg for eg in egs], key=lambda x: x["num_sample"], reverse=True)
    mixture = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(d["mix"], dtype=torch.float32) for d in dict_list],
        batch_first=True,
    )
    src = [__prepare_target_rir(dict_list, index) for index in range(num_spks)]
    input_sizes = torch.tensor([d["num_sample"] for d in dict_list], dtype=torch.float32)
    key = [d["key"] for d in dict_list]
    return input_sizes, mixture, src, key


@logger_wraps()
class MyDataset(Dataset):
    def __init__(
        self,
        max_len,
        fs,
        partition,
        data_type,
        wave_scp_srcs,
        wave_scp_mix,
        wave_scp_noise,
        dynamic_mixing,
        wave_scp_reverbs=None,
        speed_list=None,
    ):
        self.partition = partition
        self.data_type = data_type
        for wave_scp_src in wave_scp_srcs:
            if not os.path.exists(wave_scp_src):
                raise FileNotFoundError(f"Could not find file {wave_scp_src}")
        if wave_scp_reverbs:
            for wave_scp_reverb in wave_scp_reverbs:
                if not os.path.exists(wave_scp_reverb):
                    raise FileNotFoundError(f"Could not find file {wave_scp_reverb}")
        if wave_scp_noise and not os.path.exists(wave_scp_noise):
            raise FileNotFoundError(f"Could not find file {wave_scp_noise}")
        if not os.path.exists(wave_scp_mix):
            raise FileNotFoundError(f"Could not find file {wave_scp_mix}")

        self.max_len = max_len
        self.fs = fs
        self.wave_dict_srcs = [util_dataset.parse_scps(wave_scp_src) for wave_scp_src in wave_scp_srcs]
        self.wave_dict_reverbs = (
            [util_dataset.parse_scps(wave_scp_reverb) for wave_scp_reverb in wave_scp_reverbs]
            if wave_scp_reverbs
            else None
        )
        self.wave_dict_mix = util_dataset.parse_scps(wave_scp_mix)
        self.wave_dict_noise = util_dataset.parse_scps(wave_scp_noise) if wave_scp_noise else None
        self.wave_keys = list(self.wave_dict_mix.keys())
        self.dynamic_mixing = dynamic_mixing
        self.speed_list = tuple(float(factor) for factor in speed_list) if speed_list else ()
        if any(factor <= 0.0 for factor in self.speed_list):
            raise ValueError(f"Speed factors must be positive, got {self.speed_list}")
        self.noise_snr_distribution = None
        if (
            self.dynamic_mixing
            and self.wave_dict_noise is not None
            and self.data_type in NOISE_SNR_DISTRIBUTIONS
        ):
            self.noise_snr_distribution = _load_noise_snr_distribution(
                NOISE_SNR_DISTRIBUTIONS[self.data_type]
            )
        self.num_spks = len(self.wave_dict_srcs)
        self.dataset_name = {
            "clean": "wsj0-2mix",
            "noisy": "wham",
            "noisy_reverb": "whamr",
        }.get(self.data_type, self.data_type)
        key_sets = [set(d.keys()) for d in self.wave_dict_srcs]
        if self.wave_dict_reverbs:
            key_sets.extend(set(d.keys()) for d in self.wave_dict_reverbs)
        self.dynamic_key_pool = sorted(set.intersection(*key_sets)) if key_sets else []
        if self.dynamic_mixing and not self.dynamic_key_pool:
            raise ValueError("Dynamic mixing requires shared keys across speaker scp files.")
        self.dynamic_source_pool = {}
        self.dynamic_speakers = ()
        if self.dynamic_mixing:
            self._build_dynamic_source_pool()
        source_mode = "dynamic_mix" if self.dynamic_mixing else "fixed_mix"
        extra_scps = []
        if self.wave_dict_noise is not None:
            extra_scps.append("noise")
        if self.wave_dict_reverbs is not None:
            extra_scps.append("reverb_sources")
        extra_desc = ",".join(extra_scps) if extra_scps else "none"
        logger.info(
            f"Create MyDataset[{self.data_type}/{self.partition}/{source_mode}] for {wave_scp_mix} "
            f"with {len(self.wave_dict_mix)} utterances (extra_scps={extra_desc}, "
            f"speed_list={self.speed_list or 'disabled'})"
        )

    def __len__(self):
        return len(self.wave_dict_mix)

    def __contains__(self, key):
        return key in self.wave_dict_mix

    @staticmethod
    def _rms(wav):
        return np.sqrt(np.mean(np.square(wav))) + 1.0e-8

    def _sample_noise_snr(self):
        if self.noise_snr_distribution is not None:
            return _sample_empirical_noise_snr(self.noise_snr_distribution)
        return random.uniform(-3.0, 6.0)

    @staticmethod
    def _match_length(wav, len_data):
        if len(wav) == len_data:
            return wav
        if len(wav) > len_data:
            leftover = len(wav) - len_data
            idx = random.randint(0, leftover)
            return wav[idx:idx + len_data]
        pad_width = [(0, len_data - len(wav))] + [(0, 0)] * (wav.ndim - 1)
        return np.pad(wav, pad_width)

    def _match_noise_length(self, noise, len_data):
        if len(noise) >= len_data:
            return self._match_length(noise, len_data)
        if len(noise) == 0:
            raise ValueError("Noise waveform must not be empty.")

        crossfade_len = min(int(0.05 * self.fs), len(noise) // 2)
        noise_extended = noise.copy()
        while len(noise_extended) < len_data:
            if crossfade_len == 0:
                noise_extended = np.concatenate([noise_extended, noise])
                continue

            window = np.hanning(2 * crossfade_len).astype(noise.dtype)
            transition = (
                noise_extended[-crossfade_len:] * window[crossfade_len:]
                + noise[:crossfade_len] * window[:crossfade_len]
            )
            noise_extended = np.concatenate(
                [noise_extended[:-crossfade_len], transition, noise[crossfade_len:]]
            )
        return noise_extended[:len_data]

    def _speaker_id(self, key, source_idx):
        utterance_id = self._source_utterance_id(key, source_idx)
        if self.dataset_name in {"wsj0-2mix", "wham", "whamr"}:
            return utterance_id[:3]
        if self.dataset_name == "libri2mix":
            return utterance_id.split("-")[0]
        raise ValueError(f"Cannot extract speaker ID from key {key} for {self.dataset_name}.")

    def _source_utterance_id(self, key, source_idx):
        stem = os.path.splitext(os.path.basename(key))[0]
        parts = stem.split("_")
        if self.dataset_name in {"wsj0-2mix", "wham", "whamr"} and len(parts) >= 3:
            return parts[source_idx * 2]
        if self.dataset_name == "libri2mix" and len(parts) >= 2:
            return parts[source_idx]
        raise ValueError(f"Cannot extract utterance ID from key {key} for {self.dataset_name}.")

    def _build_dynamic_source_pool(self):
        source_pool = {}
        for key in self.dynamic_key_pool:
            for source_idx in range(self.num_spks):
                speaker_id = self._speaker_id(key, source_idx)
                utterance_id = self._source_utterance_id(key, source_idx)
                utterances = source_pool.setdefault(speaker_id, {})
                utterances.setdefault(utterance_id, []).append((source_idx, key))

        self.dynamic_source_pool = {
            speaker_id: tuple(
                tuple(references)
                for _, references in sorted(utterances.items())
            )
            for speaker_id, utterances in sorted(source_pool.items())
        }
        self.dynamic_speakers = tuple(self.dynamic_source_pool)
        if len(self.dynamic_speakers) < 2:
            raise ValueError("Dynamic mixing requires at least two distinct speakers.")

    def _sample_dynamic_source_refs(self):
        speakers = random.sample(self.dynamic_speakers, 2)
        return [
            random.choice(random.choice(self.dynamic_source_pool[speaker_id]))
            for speaker_id in speakers
        ]

    @staticmethod
    def _peak_guard(samps_mix, clean_sources, peak_threshold=0.9):
        max_amp = max(
            [np.max(np.abs(samps_mix))]
            + [np.max(np.abs(source)) for source in clean_sources]
        )
        if max_amp > peak_threshold:
            scale = peak_threshold / (max_amp + 1.0e-8)
            samps_mix = samps_mix * scale
            clean_sources = [source * scale for source in clean_sources]
        return samps_mix, clean_sources

    def _load_audio(self, file):
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file {file} does not exist!")
        wav, _ = audio_lib.load(file, sr=self.fs)
        return wav

    @staticmethod
    def _speed_perturb_group(waveforms, factor):
        factor = float(factor)
        if factor <= 0.0:
            raise ValueError(f"Speed factor must be positive, got {factor}")
        if factor == 1.0:
            return list(waveforms)

        common_len = min(len(waveform) for waveform in waveforms)
        stacked = np.stack(
            [waveform[:common_len] for waveform in waveforms], axis=-1
        )
        ratio = Fraction(str(factor)).limit_denominator(1000)
        perturbed = resample_poly(
            stacked,
            up=ratio.denominator,
            down=ratio.numerator,
            axis=0,
        ).astype(stacked.dtype, copy=False)
        return [perturbed[:, index] for index in range(perturbed.shape[1])]

    @classmethod
    def _speed_perturb_pair(cls, clean, mix_source, factor):
        return cls._speed_perturb_group([clean, mix_source], factor)

    def _source_file_pairs(self, key):
        if self.num_spks != 2:
            raise ValueError("Dynamic mixing currently expects two speakers.")

        source_refs = self._sample_dynamic_source_refs()
        clean_files = [
            self.wave_dict_srcs[spk_idx][src_key]
            for spk_idx, src_key in source_refs
        ]
        if self.wave_dict_reverbs:
            mix_source_files = [
                self.wave_dict_reverbs[spk_idx][src_key]
                for spk_idx, src_key in source_refs
            ]
        else:
            mix_source_files = clean_files
        return clean_files, mix_source_files

    def _dynamic_mixing(self, key):
        clean_files, mix_source_files = self._source_file_pairs(key)

        src_len = [self.max_len]
        clean_sources = []
        mix_sources = []
        ref_rms = None
        source_rel_snr = random.uniform(0.0, 5.0)
        if random.random() < 0.5:
            source_rel_snr = -source_rel_snr
        source_gains_db = [source_rel_snr / 2.0, -source_rel_snr / 2.0]

        for source_idx, (clean_file, mix_file) in enumerate(
            zip(clean_files, mix_source_files)
        ):
            clean = self._load_audio(clean_file)
            mix_source = self._load_audio(mix_file)
            if self.speed_list:
                speed_factor = random.choice(self.speed_list)
                clean, mix_source = self._speed_perturb_pair(
                    clean, mix_source, speed_factor
                )

            curr_rms = self._rms(clean)
            if ref_rms is None:
                ref_rms = curr_rms
            norm_factor = ref_rms / curr_rms
            clean *= norm_factor
            mix_source *= norm_factor

            gain = pow(10, source_gains_db[source_idx] / 20)
            clean_sources.append(gain * clean)
            mix_sources.append(gain * mix_source)
            src_len.extend([len(clean), len(mix_source)])

        noise = None
        if self.wave_dict_noise is not None:
            noise = self._load_audio(self.wave_dict_noise[key])

        min_len = min(src_len)
        paired_sources = []
        for mix_source, clean in zip(mix_sources, clean_sources):
            pair_len = min(len(mix_source), len(clean))
            pair = np.stack([mix_source[:pair_len], clean[:pair_len]], axis=-1)
            paired_sources.append(self._match_length(pair, min_len))

        mix_sources = [pair[..., 0] for pair in paired_sources]
        clean_sources = [pair[..., 1] for pair in paired_sources]
        samps_mix = sum(mix_sources)
        if noise is not None:
            noise = self._match_noise_length(noise, min_len)
            speech_rms = self._rms(samps_mix)
            target_snr = self._sample_noise_snr()
            noise *= speech_rms / self._rms(noise)
            noise *= pow(10, -target_snr / 20)
            samps_mix = samps_mix + noise

        samps_mix, clean_sources = self._peak_guard(samps_mix, clean_sources)
        samps_mix, valid_len = _pad_to_stride(samps_mix, stride=4)
        clean_sources = [_pad_to_stride(source, stride=4)[0] for source in clean_sources]
        return samps_mix, clean_sources, valid_len

    def _direct_load(self, key):
        samps_src = []
        files = [self.wave_dict_srcs[0][key], self.wave_dict_srcs[1][key]]
        for file in files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} do not exists!")
            samps_tmp, _ = audio_lib.load(file, sr=self.fs)
            samps_src.append(samps_tmp)

        libri_independent_speed = (
            self.partition == "train"
            and self.dataset_name == "libri2mix"
            and self.speed_list
        )
        if libri_independent_speed:
            samps_src = [
                self._speed_perturb_group(
                    [source], random.choice(self.speed_list)
                )[0]
                for source in samps_src
            ]
            mix_len = min(len(source) for source in samps_src)
            samps_src = [self._match_length(source, mix_len) for source in samps_src]
            samps_mix = np.sum(np.stack(samps_src, axis=0), axis=0)
            samps_mix, samps_src = self._peak_guard(samps_mix, samps_src)
        else:
            file = self.wave_dict_mix[key]
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} do not exists!")
            samps_mix, _ = audio_lib.load(file, sr=self.fs)

            if self.speed_list:
                speed_factor = random.choice(self.speed_list)
                perturbed = self._speed_perturb_group(
                    [samps_mix, *samps_src], speed_factor
                )
                samps_mix, samps_src = perturbed[0], perturbed[1:]

        if len(samps_mix) % 4 != 0:
            remains = len(samps_mix) % 4
            samps_mix = samps_mix[:-remains]
            samps_src = [s[:-remains] for s in samps_src]

        if len(samps_mix) > self.max_len:
            if self.partition == "train":
                start = random.randint(0, len(samps_mix) - self.max_len)
            elif self.partition == "valid":
                start = (len(samps_mix) - self.max_len) // 2
            else:
                start = None
            if start is not None:
                samps_mix = samps_mix[start:start + self.max_len]
                samps_src = [s[start:start + self.max_len] for s in samps_src]

        return samps_mix, samps_src

    def __getitem__(self, index):
        key = self.wave_keys[index]
        if any(key not in self.wave_dict_srcs[i] for i in range(len(self.wave_dict_srcs))) or key not in self.wave_dict_mix:
            raise KeyError(f"Could not find utterance {key}")
        if self.wave_dict_reverbs is not None and any(key not in self.wave_dict_reverbs[i] for i in range(len(self.wave_dict_reverbs))):
            raise KeyError(f"Could not find reverb utterance {key}")

        if self.dynamic_mixing:
            samps_mix, samps_src, valid_len = self._dynamic_mixing(key)
        else:
            samps_mix, samps_src = self._direct_load(key)
            valid_len = samps_mix.shape[0]
        if self.partition in {"train", "valid"} and samps_mix.shape[0] < self.max_len:
            pad_len = self.max_len - samps_mix.shape[0]
            samps_mix = np.pad(samps_mix, (0, pad_len))
            samps_src = [np.pad(source, (0, pad_len)) for source in samps_src]
        return {"num_sample": valid_len, "mix": samps_mix, "src": samps_src, "key": key}
