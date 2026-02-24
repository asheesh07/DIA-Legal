from dataclasses import dataclass
from datetime import datetime
from pytube import YouTube
from pathlib import Path
import uuid
import shutil
import yt_dlp
import os
@dataclass
class VideoAsset:
    evidence_id: str
    case_id: str
    source_type:str
    original_name:str
    stored_path:str
    created_at : datetime
    status : str

class BaseReader:
    
    def validate(self,source:str)->None:
        raise NotImplementedError
        
    def load(self,source:str,case_id:str) -> VideoAsset:
        raise NotImplementedError
    
class YTReader(BaseReader):
    def validate(self, source):
        if "youtube.com" not in source and "youtu.be" not in source:
            raise ValueError("Invalid Url Error")
    
    def load(self, source, case_id,output_path):
        evidence_id = str(uuid.uuid4())
        case_folder = Path(output_path) / case_id / "raw"
        case_folder.mkdir(parents = True , exist_ok = True )   
        stored_path,title = self._download_yt_video(source,output_path)
        asset =VideoAsset(
            evidence_id=evidence_id,
            case_id=case_id,
            source_type="Youtube",
            original_name=title,
            stored_path=stored_path,
            created_at= datetime.utcnow(),
            status= "stored"
        )
        
        return asset
    def _download_yt_video(self,url,output_path):
        
        ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
    }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = str(ydl.prepare_filename(info))
            title = info.get('title','UNKNOWN')

        return file_path,title
class LocalReader(BaseReader):
    Allowed_extensions ={".mp4", ".mov", ".mkv", ".avi", ".webm"}
    def validate(self, source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError("Video file does not exists")
        if path.suffix.lower() not in self.Allowed_extensions:
            raise ValueError(f"Unsupportive video file format{path.suffix} ")
        
    
    def load(self, source, case_id,output_path):
        evidence_id =str(uuid.uuid4())
        source_path =Path(source)
        case_folder = Path(output_path)/case_id/"raw"
        case_folder.mkdir(parents=True,exists_ok = True)
        filename = f"{evidence_id}{source_path.suffix}"
        final_path = case_folder /filename
        shutil.copy2(source_path,final_path)
        asset = VideoAsset(
            evidence_id=evidence_id,
            case_id=case_id,
            source_type="Local",
            original_name= source_path.name,
            stored_path=str(final_path),
            created_at=datetime.utcnow(),
            status="Stored"
        )
        return asset

class ReaderRouter:
    def from_path(source):
        if source.startswith('https://') or source.startswith("http://"):
            if "youtube.com" in source or "youtu.be" in source:
                return YTReader()
            else:
                raise ValueError('unsupported url')
            
        path = Path(source)
        if path.exists():
            return LocalReader
        
        raise ValueError('unsupported type')
        
    
    