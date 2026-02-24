import subprocess  
import whisperx
import torch 
import cv2      
import uuid 
import pytesseract 
from whisperx.diarize import DiarizationPipeline
from transformers import AutoProcessor,AutoModelForVision2Seq
from pathlib import Path
from PIL import Image    
import os
HF_TOKEN = os.getenv("HF_TOKEN")                                                                            
class VideoProcessor:
    def __init__(self,base_output_path,model_size:str ="base"):
        self.base_output_path = Path(base_output_path)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.dtype = 'float16' if self.device =='cuda' else 'float32'
        self.model =whisperx.load_model(model_size,self.device,compute_type =self.dtype)  
        self.diarize_model = DiarizationPipeline(
            token=HF_TOKEN,
            device=self.device
        )  
        self.processor = AutoProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.caption_model = AutoModelForVision2Seq.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.align_model = None
        self.align_metadata = None
    def process(self,asset):
        audio_path = self.video_to_audio(asset)
        transcript =self.audio_to_text(audio_path)
        frames = self.video_to_images(asset) # i have to add strategy and interval 
        analysed_frames = self.analyse_frames(frames)
        aligned_data =self.aligned_modalities(analysed_frames,transcript,asset.case_id)
        return aligned_data
    
    def video_to_audio(self, asset):

        audio_dir = self.base_output_path / asset.case_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        output_audio_path = audio_dir / f"{asset.evidence_id}.wav"

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i", str(asset.stored_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",
            str(output_audio_path)
        ]

        try:
            result = subprocess.run(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg failed: {e.stderr.decode()}"
            )

        if not output_audio_path.exists():
            raise RuntimeError("Audio extraction failed, output file not created")

        return str(output_audio_path)

    
    def audio_to_text(self, audio_path):

        result = self.model.transcribe(audio_path)

        language = result.get("language", "en")

        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device
        )

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_path,
            self.device
        )
        diarize_segments = self.diarize_model(audio_path)

        result = whisperx.assign_word_speakers(
            diarize_segments,
            result
        )
        segments = []

        for seg in result["segments"]:
            segments.append({
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg["text"],
                "speaker": seg.get("speaker", "UNKNOWN")
            })

        return segments

     
    def video_to_images(self,asset,strategy = "Hybrid",interval_sec:int =1):
        path = asset.stored_path
        frames_dir = self.base_output_path / asset.case_id / "frames" / asset.evidence_id
        frames_dir.mkdir(parents=True,exist_ok = True)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Failed to load the video")
        fps =cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps*interval_sec)
        frames_metadata =[]
        frame_count = 0
        saved_count = 0
        while True:
            ret , frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp =frame_count /fps
                frame_filename = f"{saved_count:04d}_{timestamp:.3f}.jpg"
                frame_path = frames_dir/frame_filename
                cv2.imwrite(str(frame_path),frame)
                
                frames_metadata =[{
                    "frame_id":str(uuid.uuid4()),
                    "timestamp":timestamp,
                    "image_path":frame_path
                }]
                saved_count+=1
            frame_count+=1
        cap.release()
        
        return frames_metadata
    
    def analyse_frames(self,frames_metadata):
        analysed_frames =[]
        for frame in frames_metadata:
            ocr_text =self.extract_text(frame['image_path'])
            caption = self.generate_captions(frame['image_path'])
            analysed_frames.append({
                "frame_id": frame['frame_id'],
                "timestamp":frame['timestamp'],
                "image_path":frame['image_path'],
                "ocr_text": ocr_text,
                "caption":caption
            })
        
        return analysed_frames
    
    def extract_text(self,path):
        image = cv2.imread(path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    
    def generate_captions(self,path):
        image = Image.open(path).convert("RGB")
        inputs = self.processor(image,return_tensors="pt")
        out = self.caption_model.generate(**inputs)
        caption = self.processor.decode(out[0],skip_special_tokens=True)
        return caption
    
    def aligned_modalities(self,frames,transcript,case_id,buffer = 0.5):
        aligned_chunks = []
        frames = sorted(frames,key=lambda x:x['timestamp'])
        for segment in transcript:
            seg_start = segment['start_time']
            seg_end =segment['end_time']
            relevant_frames = [frame for frame in frames if (seg_start-buffer) <= frame['timestamp'] <= (seg_end + buffer)]
            aligned_chunks.append({
                "case_id":case_id,
                "start":seg_start,
                "end":seg_end,
                "transcript":segment['text'],
                'speaker':segment['speaker'],
                'frames':relevant_frames
                
            })
        
        return aligned_chunks
    