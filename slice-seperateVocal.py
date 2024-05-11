from tkinter import *
from tkinter import filedialog
import torch
import os
import librosa
import math
import soundfile
import numpy as np

class AudioProcessor:
    def __init__(self, sr: int, threshold: float = -40., min_length: int = 5000, min_interval: int = 300, hop_size: int = 20, max_sil_kept: int = 5000):
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(sr * min_interval / 1000), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(sr * min_interval / 1000 / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _get_rms(self, y, frame_length=2048, hop_length=512, pad_mode="constant"):
        padding = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode=pad_mode)
        axis = -1
        out_strides = y.strides + tuple([y.strides[axis]])
        x_shape_trimmed = list(y.shape)
        x_shape_trimmed[axis] -= frame_length - 1
        out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
        xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
        if axis < 0:
            target_axis = axis - 1
        else:
            target_axis = axis + 1
        xw = np.moveaxis(xw, -1, target_axis)
        slices = [slice(None)] * xw.ndim
        slices[axis] = slice(0, None, hop_length)
        x = xw[tuple(slices)]
        power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
        return np.sqrt(power)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = self._get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

class AudioSplitter:
    def __init__(self, window):
        self.window = window
        self.file_text = StringVar()
        self.listbox = Listbox(self.window, height=6, width=35)
        self.scrollbar = Scrollbar(self.window)

    def split_audio(self):
        audio_path = self.file_text.get()
        if not os.path.isfile(audio_path):
            print(f"File {audio_path} does not exist.")
            return
        audio, sr = librosa.load(audio_path, sr=None, mono=False)
        processor = AudioProcessor(sr=sr)
        chunks = processor.slice(audio)
        output_dir = os.path.join(os.path.dirname(self.file_text.get()), "split_audio")
        os.makedirs(output_dir, exist_ok=True)
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            soundfile.write(os.path.join(output_dir, f'{os.path.basename(audio_path).replace(" ", "_")}_split_{i}.wav'), chunk, sr)
        self.listbox.insert(END, f"Split Audio: {output_dir}")
        return output_dir




    def separate_vocals(self):
        audio_path = self.file_text.get()
        if not os.path.isfile(audio_path):
            print(f"File {audio_path} does not exist.")
            return
        output_dir = os.path.join(os.path.dirname(self.file_text.get()), "split_audio")
        os.makedirs(output_dir, exist_ok=True)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        # Calculate chunk size for 20 minutes
        chunk_size = sr * 60 * 20  # 20 minutes

        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Limit the number of cores to 8 if using CPU
        if device == 'cpu':
            os.environ["OMP_NUM_THREADS"] = "8"
            os.environ["MKL_NUM_THREADS"] = "8"

        # Split audio into chunks and process each chunk with Demucs
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk.shape) > 1:
                chunk = chunk.T
            temp_audio_path = os.path.join(output_dir, f'{os.path.basename(audio_path).replace(" ", "_")}_temp_{math.floor(i/chunk_size)}.wav')
            soundfile.write(temp_audio_path, chunk, sr)
            os.system(f"demucs -d {device} --two-stems=vocals \"{temp_audio_path}\" -o \"{output_dir}\"")
            os.remove(temp_audio_path)  # remove temporary file after processing

        self.listbox.insert(END, f"Separate Vocals: {output_dir}")
        return output_dir


    def create_gui(self):
        Label(self.window, text="FIle Name").grid(row=0, column=0)
        Entry(self.window, textvariable=self.file_text).grid(row=0, column=1)
        self.listbox.grid(row=2, column=0, rowspan=6, columnspan=2)
        self.scrollbar.grid(row=2, column=2, rowspan=6)
        self.listbox.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.configure(command=self.listbox.yview)
        Button(self.window, text="Find File", width=12, command=lambda: self.file_text.set(filedialog.askopenfilename())).grid(row=3, column=3)
        Button(self.window, text="Separate Vocals", width=12, command=self.separate_vocals).grid(row=4, column=3)
        Button(self.window, text="Split Audio", width=12, command=self.split_audio).grid(row=5, column=3)
        Button(self.window, text="Close", width=12, command=self.window.quit).grid(row=6, column=3)

if __name__ == "__main__":
    window = Tk()
    splitter = AudioSplitter(window)
    splitter.create_gui()
    window.mainloop()


    