import subprocess
import torch
import uuid
import pytesseract
from pathlib import Path
from PIL import Image
import os
import concurrent.futures

HF_TOKEN = os.getenv("HF_TOKEN")

_STREAM_PREFIXES = ("http://", "https://", "rtmp://", "rtsp://")


def _is_stream(path: str) -> bool:
    return isinstance(path, str) and path.startswith(_STREAM_PREFIXES)


class VideoProcessor:
    def __init__(self, base_output_path, model_size: str = "base"):
        self.base_output_path = Path(base_output_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = "float16" if self.device == "cuda" else "float32"
        self.model_size = model_size

        self._model = None
        self._diarize_model = None
        self._processor = None
        self._caption_model = None

    # ── Lazy properties ──────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            import whisperx
            print("[VideoProcessor] Loading WhisperX model...", flush=True)
            self._model = whisperx.load_model(
                self.model_size, self.device, compute_type=self.dtype
            )
        return self._model

    @property
    def diarize_model(self):
        if self._diarize_model is None:
            from whisperx.diarize import DiarizationPipeline
            print("[VideoProcessor] Loading diarization model...", flush=True)
            self._diarize_model = DiarizationPipeline(
                token=HF_TOKEN, device=self.device
            )
        return self._diarize_model

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor
            print("[VideoProcessor] Loading BLIP processor...", flush=True)
            self._processor = AutoProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        return self._processor

    @property
    def caption_model(self):
        if self._caption_model is None:
            from transformers import AutoModelForImageTextToText
            print("[VideoProcessor] Loading BLIP caption model...", flush=True)
            self._caption_model = AutoModelForImageTextToText.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
        return self._caption_model

    # ── Public API ────────────────────────────────────────────────

    def process(self, asset, progress=None):
        import time as _time
        _t_total = _time.time()

        def emit(stage):
            if progress:
                progress(stage)

        path = str(asset.stored_path)

        if _is_stream(path):
            emit("extracting_audio")
            _t1 = _time.time()
            audio_path = self.video_to_audio(asset)
            emit("sampling_frames")
            frames = self.video_to_images(asset)
            print(f"[TIMING][VIDEO] ffmpeg (stream): {_time.time()-_t1:.2f}s", flush=True)
        else:
            emit("extracting_audio")
            _t1 = _time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                audio_fut = pool.submit(self.video_to_audio, asset)
                frame_fut = pool.submit(self.video_to_images, asset)
                audio_path = audio_fut.result()
                frames = frame_fut.result()
            print(f"[TIMING][VIDEO] ffmpeg audio+frames (parallel): {_time.time()-_t1:.2f}s  frames={len(frames)}", flush=True)

        emit("transcribing")
        _t2 = _time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            transcript_fut = pool.submit(self.audio_to_text, audio_path)
            frames_fut     = pool.submit(self.analyse_frames, frames)
            transcript = transcript_fut.result()
            analysed   = frames_fut.result()
        print(f"[TIMING][VIDEO] transcribe+OCR (parallel): {_time.time()-_t2:.2f}s  segments={len(transcript)}", flush=True)

        emit("chunking")
        result = self.aligned_modalities(analysed, transcript, asset.case_id)
        print(f"[TIMING][VIDEO] TOTAL process(): {_time.time()-_t_total:.2f}s", flush=True)
        return result

    def process_from_files(self, audio_path: str, frames_dir: Path,
                           case_id: str, evidence_id: str, progress=None):
        """
        ML-only pipeline used when FFmpeg has already produced audio + frames
        (streaming upload path — no asset object needed).
        """
        def emit(stage):
            if progress:
                progress(stage)

        frames_metadata = self._collect_frames(frames_dir, interval_sec=60)

        emit("transcribing")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            transcript_fut = pool.submit(self.audio_to_text, audio_path)
            frames_fut     = pool.submit(self.analyse_frames, frames_metadata)
            transcript = transcript_fut.result()
            analysed   = frames_fut.result()
        emit("chunking")
        return self.aligned_modalities(analysed, transcript, case_id)

    # ── Audio ─────────────────────────────────────────────────────

    def video_to_audio(self, asset):
        audio_dir = self.base_output_path / asset.case_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        out = audio_dir / f"{asset.evidence_id}.wav"

        cmd = [
            "ffmpeg", "-y", "-i", str(asset.stored_path),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg audio failed: {e.stderr.decode()}")

        if not out.exists():
            raise RuntimeError("Audio extraction produced no output")
        return str(out)

    # ── Frames ────────────────────────────────────────────────────

    def video_to_images(self, asset, interval_sec: int = 60):
        path = str(asset.stored_path)
        frames_dir = (
            self.base_output_path / asset.case_id / "frames" / asset.evidence_id
        )
        frames_dir.mkdir(parents=True, exist_ok=True)

        if _is_stream(path):
            return self._frames_from_stream(path, frames_dir, interval_sec)
        return self._frames_from_local(path, frames_dir, interval_sec)

    def _frames_from_stream(self, stream_url: str, frames_dir: Path, interval_sec: int):
        """Sample 1 frame every N seconds from a stream URL via FFmpeg thumbnail filter."""
        pattern = str(frames_dir / "%04d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", stream_url,
            "-vf", f"thumbnail,fps=1/{interval_sec}",
            "-vsync", "vfr",
            pattern,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"[VideoProcessor] Stream frame extraction failed: "
                f"{e.stderr.decode()[:200]}",
                flush=True,
            )
            return []
        return self._collect_frames(frames_dir, interval_sec)

    def _frames_from_local(self, video_path: str, frames_dir: Path, interval_sec: int):
        """Extract scene-change frames at 640px width from a local file."""
        pattern = str(frames_dir / "%04d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", "select='gt(scene,0.3)',scale=640:-1",
            "-vsync", "vfr",
            pattern,
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"[VideoProcessor] Local frame extraction failed: "
                f"{e.stderr.decode()[:200]}",
                flush=True,
            )
            return []
        return self._collect_frames(frames_dir, interval_sec)

    def _collect_frames(self, frames_dir: Path, interval_sec: int) -> list:
        jpegs = sorted(frames_dir.glob("*.jpg"))
        return [
            {
                "frame_id":   str(uuid.uuid4()),
                "timestamp":  float(i * interval_sec),
                "image_path": jpg,
            }
            for i, jpg in enumerate(jpegs)
        ]

    # ── Transcription ─────────────────────────────────────────────

    @staticmethod
    def _format_ts(sec: float) -> str:
        sec = max(0, int(sec))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _map_speakers(segments: list) -> dict:
        seen: dict = {}
        roles = ["Q", "A", "W1", "W2", "W3"]
        for seg in segments:
            sp = seg.get("speaker", "UNKNOWN")
            if sp and sp not in ("UNKNOWN", "") and sp not in seen:
                idx = len(seen)
                seen[sp] = roles[idx] if idx < len(roles) else sp
        return seen

    def audio_to_text(self, audio_path):
        import whisperx, time as _time
        _t0 = _time.time()

        result = self.model.transcribe(audio_path)
        language = result.get("language", "en")
        print(f"[TIMING][VIDEO] whisper transcribe: {_time.time()-_t0:.2f}s  lang={language}", flush=True)

        _t1 = _time.time()
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=self.device
            )
            result = whisperx.align(
                result["segments"], model_a, metadata, audio_path, self.device
            )
            print(f"[TIMING][VIDEO] alignment: {_time.time()-_t1:.2f}s", flush=True)
        except Exception as exc:
            print(f"[VideoProcessor] Alignment skipped: {exc}", flush=True)

        _t2 = _time.time()
        try:
            diarize_segments = self.diarize_model(audio_path)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"[TIMING][VIDEO] diarization: {_time.time()-_t2:.2f}s", flush=True)
        except Exception as exc:
            print(f"[VideoProcessor] Diarization skipped: {exc}", flush=True)

        raw = [
            {
                "start_time": seg["start"],
                "end_time":   seg["end"],
                "text":       seg["text"],
                "speaker":    seg.get("speaker", "UNKNOWN"),
            }
            for seg in result["segments"]
        ]
        speaker_map = self._map_speakers(raw)
        return [
            {
                "start_time":      s["start_time"],
                "end_time":        s["end_time"],
                "text":            s["text"],
                "speaker":         speaker_map.get(s["speaker"], s["speaker"]),
                "timestamp_start": self._format_ts(s["start_time"]),
                "timestamp_end":   self._format_ts(s["end_time"]),
            }
            for s in raw
        ]

    # ── Frame analysis ────────────────────────────────────────────

    def analyse_frames(self, frames_metadata):
        import time as _time
        _t0 = _time.time()
        # BLIP captioning is skipped on CPU — 600 MB model at ~3s/frame is
        # the dominant per-frame cost and adds little over OCR for legal docs.
        for f in frames_metadata:
            f["caption"]  = ""
            f["ocr_text"] = self._ocr(f["image_path"])
        print(f"[TIMING][VIDEO] OCR {len(frames_metadata)} frames: {_time.time()-_t0:.2f}s", flush=True)
        return frames_metadata

    def _ocr(self, path):
        import cv2
        img = cv2.imread(str(path))
        return pytesseract.image_to_string(img).strip() if img is not None else ""

    # ── Alignment ─────────────────────────────────────────────────

    def aligned_modalities(self, frames, transcript, case_id, buffer=0.5):
        frames = sorted(frames, key=lambda x: x["timestamp"])
        return [
            {
                "case_id":         case_id,
                "start":           seg["start_time"],
                "end":             seg["end_time"],
                "transcript":      seg["text"],
                "speaker":         seg["speaker"],
                "timestamp_start": seg.get("timestamp_start", ""),
                "timestamp_end":   seg.get("timestamp_end", ""),
                "frames": [
                    f for f in frames
                    if (seg["start_time"] - buffer) <= f["timestamp"] <= (seg["end_time"] + buffer)
                ],
            }
            for seg in transcript
        ]
