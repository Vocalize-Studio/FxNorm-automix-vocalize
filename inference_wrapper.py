
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import importlib.util
import time
from typing import Optional, Dict, List, Callable, Any

from automix.common_supernet import SuperNet
from automix.data_normalization import DataNormalizer
from automix.common_miscellaneous import uprint, get_network_and_supernet, load_model, pad_to_shape

class Progress:
    """A simple dataclass to report progress."""
    def __init__(self, pct: float, status: str, meta: Optional[Dict[str, Any]] = None):
        self.pct = pct
        self.status = status
        self.meta = meta or {}

class FxNormAutomixWrapper:
    """
    A wrapper for the FxNorm-automix inference process, providing a clean,
    programmatic interface to mix audio stems.
    """
    def __init__(self, config_path: str, model_path: str, device: Optional[str] = None):
        """
        Initialize the FxNorm-automix wrapper.

        Args:
            config_path (str): Path to the model's Python configuration file.
            model_path (str): Path to the pre-trained model parameters file (.params).
            device (Optional[str]): The device to run the model on (e.g., 'cuda', 'cpu').
                                    If None, it will be auto-detected.
        """
        uprint(f"Loading configuration from {config_path}")
        self.config = self._load_config_from_path(config_path)
        self.model_path = model_path

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        uprint(f"Using device: {self.device}")

        # Load model and normalizer
        self._load_model()

    def _load_config_from_path(self, path: str):
        """Dynamically load a Python module from a file path."""
        path = Path(path)
        spec = importlib.util.spec_from_file_location(path.stem, path.resolve())
        if spec is None:
            raise ImportError(f"Could not load spec for module from path {path}")
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module

    def _load_model(self):
        """Load the network, supernet, and data normalizer."""
        # Build the network and supernet
        self.net, self.super_net = get_network_and_supernet(self.config)
        self.super_net.to(self.device)

        # Load model weights
        load_model(self.super_net, self.model_path, self.device)
        self.super_net.eval()

        # Initialize data normalizer
        self.normalizer = DataNormalizer(self.config.TARGET_LEVEL, self.config.FX_CHAIN, self.config.SAMPLE_RATE)

    def _load_and_prepare_stem(self, path: Optional[str], sr: int, channels: int) -> np.ndarray:
        """
        Load an audio file, resample it, and convert it to the correct channel format.
        Returns a silent array if the path is None.
        """
        if path is None or not Path(path).exists():
            return np.zeros((0, channels), dtype=np.float32)

        try:
            audio, file_sr = librosa.load(path, sr=None, mono=False)
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)
            audio = audio.T # (samples, channels)
        except Exception as e:
            uprint(f"Could not load audio file {path}: {e}")
            return np.zeros((0, channels), dtype=np.float32)

        # Resample if necessary
        if file_sr != sr:
            audio = librosa.resample(audio.T, orig_sr=file_sr, target_sr=sr).T

        # Adjust channels
        if audio.shape[1] != channels:
            if audio.shape[1] == 1 and channels == 2:
                uprint(f"Converted file {Path(path).name} to stereo by repeating mono channel")
                audio = np.concatenate([audio, audio], axis=1)
            else:
                uprint(f"Warning: Channel mismatch for {Path(path).name}. Expected {channels}, got {audio.shape[1]}. Truncating/padding.")
                new_audio = np.zeros((audio.shape[0], channels), dtype=audio.dtype)
                min_channels = min(audio.shape[1], channels)
                new_audio[:, :min_channels] = audio[:, :min_channels]
                audio = new_audio
        
        return audio

    @torch.inference_mode()
    def mix(self, output_path: str, stem_paths: Dict[str, Optional[str]], progress_cb: Optional[Callable[[Progress], None]] = None):
        """
        Create a mix from the provided audio stems.

        Args:
            output_path (str): Path to save the final mixed .wav file.
            stem_paths (Dict[str, Optional[str]]): A dictionary mapping stem names
                (e.g., 'vocals', 'bass') to their file paths. Stems not provided
                will be treated as silence.
            progress_cb (Optional[Callable]): A callback to report progress.
        """
        def _emit(pct, status, meta=None):
            if progress_cb:
                progress_cb(Progress(pct=pct, status=status, meta=meta))

        t_start = time.time()
        _emit(0.01, "Starting...")

        _emit(0.05, "Loading and preparing stems...")
        target_sr = self.config.SAMPLE_RATE
        target_channels = self.config.N_CHANNELS
        stems = {s: self._load_and_prepare_stem(p, target_sr, target_channels) for s, p in stem_paths.items()}

        # Find max length and pad all stems to match
        max_len = max(s.shape[0] for s in stems.values()) if any(s.shape[0] > 0 for s in stems.values()) else 0
        if max_len == 0:
            raise ValueError("All input stems are empty or could not be loaded.")

        for name, audio in stems.items():
            if audio.shape[0] < max_len:
                uprint(f"{name} stem does not have same size as the rest, zero padding...")
                stems[name] = pad_to_shape(audio, max_len)

        _emit(0.25, "Normalizing stems...")
        norm_stems_np = [self.normalizer.process(stems[s_name], s_name) for s_name in self.config.STEM_ORDER]
        
        _emit(0.50, "Preparing tensor for model...")
        norm_stems_torch = [torch.from_numpy(s).to(self.device) for s in norm_stems_np]
        input_tensor = torch.stack(norm_stems_torch, dim=0).unsqueeze(0)

        _emit(0.60, "Applying model for mixing...")
        output_mix_tensor = self.super_net.inference(input_tensor)

        _emit(0.90, "Finalizing audio...")
        output_audio_tensor = output_mix_tensor[self.super_net.net.output_type].squeeze(0)
        output_audio_np = output_audio_tensor.cpu().numpy()

        uprint(f"Saving mixed audio to {output_path}...")
        sf.write(output_path, output_audio_np, target_sr)

        t_end = time.time()
        duration_sec = output_audio_np.shape[0] / target_sr
        total_time = t_end - t_start
        uprint(f"--- It took {total_time:.2f} seconds ---")
        uprint(f"--- to mix {duration_sec:.2f} seconds ---")
        _emit(1.0, "Done", meta={"total_time_s": total_time, "duration_s": duration_sec})


if __name__ == '__main__':
    # This is an example of how to use the wrapper.
    # Users should modify the paths according to their project structure.
    
    # --- Configuration ---
    # IMPORTANT: Adjust these paths before running!
    CONFIG_PATH = './configs/ISMIR/ours_S_Lb.py'
    MODEL_PATH = './trainings/results/ours_S_Lb/net_mixture.dump' # Example path
    OUTPUT_PATH = './mixed_output_from_wrapper.wav'

    # --- Input Stems ---
    # Provide paths to the audio files for each stem.
    # If a stem is not available, set its path to None.
    STEM_PATHS = {
        'vocals': './mixes/vocals/IF_VOCALS.wav',
        'drums': None, # Example of a missing stem
        'bass': None,  # Example of a missing stem
        'other': './mixes/instrumental/IF_INSTRUMENTAL.wav',
    }

    # --- Create and Run Wrapper ---
    try:
        # Example progress callback
        def my_progress_handler(p: Progress):
            print(f"[PROGRESS] {int(p.pct * 100)}% - {p.status}")

        mixer = FxNormAutomixWrapper(config_path=CONFIG_PATH, model_path=MODEL_PATH)
        
        # Create a dictionary with stems ordered according to the model's config
        ordered_stems = {stem_name: STEM_PATHS.get(stem_name) for stem_name in mixer.config.STEM_ORDER}

        mixer.mix(
            output_path=OUTPUT_PATH,
            stem_paths=ordered_stems,
            progress_cb=my_progress_handler
        )
        uprint(f"\nSuccessfully created mix and saved to {OUTPUT_PATH}")
    except Exception as e:
        uprint(f"An error occurred: {e}")
        uprint("Please ensure that the config, model, and stem paths are correct.")
