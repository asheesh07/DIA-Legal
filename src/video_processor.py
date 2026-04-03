import subprocess  
import torch 
import cv2      
import uuid 
import pytesseract 
from pathlib import Path
from PIL import Image    
import os

HF_TOKEN = os.getenv("HF_TOKEN")


class VideoProcessor:
    def __init__(self, base_output_path, model_size: str = "base"):
        self.base_output_path = Path(base_output_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = "float16" if self.device == "cuda" else "float32"
        self.model_size = model_size

        # Lazy — not loaded until first use
        self._model = None
        self._diarize_model = None
        self._processor = None
        self._caption_model = None
        self.align_model = None
        self.align_metadata = None

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

    def process(self, asset):
        audio_path    = self.video_to_audio(asset)
        transcript    = self.audio_to_text(audio_path)
        frames        = self.video_to_images(asset)
        analysed      = self.analyse_frames(frames)
        aligned_data  = self.aligned_modalities(analysed, transcript, asset.case_id)
        return aligned_data

    def video_to_audio(self, asset):
        audio_dir = self.base_output_path / asset.case_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_audio_path = audio_dir / f"{asset.evidence_id}.wav"

        ffmpeg_command = [
            "ffmpeg", "-y", "-i", str(asset.stored_path),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
            str(output_audio_path)
        ]
        try:
            subprocess.run(ffmpeg_command, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")

        if not output_audio_path.exists():
            raise RuntimeError("Audio extraction failed")
        return str(output_audio_path)

    def audio_to_text(self, audio_path):
        import whisperx
        result   = self.model.transcribe(audio_path)
        language = result.get("language", "en")

        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio_path, self.device
        )
        diarize_segments = self.diarize_model(audio_path)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        return [
            {
                "start_time": seg["start"],
                "end_time":   seg["end"],
                "text":       seg["text"],
                "speaker":    seg.get("speaker", "UNKNOWN"),
            }
            for seg in result["segments"]
        ]

    def video_to_images(self, asset, strategy="Hybrid", interval_sec: int = 5):
        path       = asset.stored_path
        frames_dir = self.base_output_path / asset.case_id / "frames" / asset.evidence_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Failed to load the video")

        fps            = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_sec)
        frames_metadata = []
        frame_count = saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp      = frame_count / fps
                frame_filename = f"{saved_count:04d}_{timestamp:.3f}.jpg"
                frame_path     = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                frames_metadata = [{
                    "frame_id":   str(uuid.uuid4()),
                    "timestamp":  timestamp,
                    "image_path": frame_path,
                }]
                saved_count += 1
            frame_count += 1

        cap.release()
        return frames_metadata

    def analyse_frames(self, frames_metadata):
        return [
            {
                "frame_id":   frame["frame_id"],
                "timestamp":  frame["timestamp"],
                "image_path": frame["image_path"],
                "ocr_text":   self.extract_text(frame["image_path"]),
                "caption":    self.generate_captions(frame["image_path"]),
            }
            for frame in frames_metadata
        ]

    def extract_text(self, path):
        image = cv2.imread(str(path))
        return pytesseract.image_to_string(image).strip()

    def generate_captions(self, path):
        image  = Image.open(path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        out    = self.caption_model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def aligned_modalities(self, frames, transcript, case_id, buffer=0.5):
        frames = sorted(frames, key=lambda x: x["timestamp"])
        return [
            {
                "case_id":   case_id,
                "start":     seg["start_time"],
                "end":       seg["end_time"],
                "transcript": seg["text"],
                "speaker":   seg["speaker"],
                "frames": [
                    f for f in frames
                    if (seg["start_time"] - buffer) <= f["timestamp"] <= (seg["end_time"] + buffer)
                ],
            }
            for seg in transcript
        ]