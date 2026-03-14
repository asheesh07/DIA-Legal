from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid
import shutil
import yt_dlp
import os
import uuid
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import pdfplumber

@dataclass
class VideoAsset:
    evidence_id: str
    case_id: str
    source_type:str
    original_name:str
    stored_path:str
    created_at : datetime
    status : str

@dataclass
class PageSpan:
    start_page: int
    end_page: int


@dataclass
class LegalSection:
    section_id: str
    title: str
    text: str
    page_span: PageSpan
    section_type: str       
    word_count: int


@dataclass
class PDFAsset:
    evidence_id: str
    case_id: str
    source_type: str        
    
    original_name: str
    stored_path: str
    created_at: datetime
    status: str

    total_pages: int
    total_words: int
    sections: List[LegalSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

_SECTION_PATTERNS = [
    # Numbered headings:  "1."  "1.1"  "2.3.1"
    (r'^\s*(\d+(?:\.\d+)*)\s*[.)]\s+[A-Z]', "numbered"),
    # ALL CAPS headings:  "STATEMENT OF FACTS"
    (r'^\s*[A-Z][A-Z\s]{4,}$',              "caps"),
    # Legal keyword headings
    (r'^\s*(WHEREAS|THEREFORE|WHEREFORE|'
     r'IN THE MATTER|BEFORE THE|'
     r'EXAMINATION|CROSS.EXAMINATION|'
     r'RE.EXAMINATION|EXHIBIT\s+[A-Z0-9]+|'
     r'SCHEDULE|ANNEXURE|APPENDIX)',         "legal_keyword"),
]

_SECTION_TYPE_MAP = {
    "statement of facts":   "facts",
    "facts":                "facts",
    "background":           "background",
    "evidence":             "evidence",
    "examination":          "examination",
    "cross examination":    "cross_examination",
    "cross-examination":    "cross_examination",
    "re-examination":       "re_examination",
    "exhibit":              "exhibit",
    "order":                "order",
    "judgment":             "order",
    "decree":               "order",
    "charge":               "charge",
    "allegations":          "charge",
    "relief":               "relief",
    "prayer":               "relief",
    "annexure":             "annexure",
    "schedule":             "schedule",
}

_DOC_TYPE_PATTERNS = [
    ("fir",               [r"\bfir\b", r"first information report",
                           r"station house officer", r"fir no"]),
    ("witness_statement", [r"witness statement", r"deposition",
                           r"affidavit", r"i\s+hereby\s+state"]),
    ("court_order",       [r"it is hereby ordered", r"court order",
                           r"judgment", r"decree", r"tribunal"]),
    ("evidence",          [r"exhibit\s+[a-z0-9]+", r"forensic",
                           r"post.?mortem", r"laboratory report"]),
    ("charge_sheet",      [r"charge\s?sheet", r"chargesheet",
                           r"accused", r"sections?\s+\d+"]),
]

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
        case_folder.mkdir(parents=True,exist_ok = True)
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

class PDFReader(BaseReader):

    ALLOWED_EXTENSIONS = {".pdf"}

    def validate(self, source: str) -> None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. Expected .pdf"
            )

    def load(self, source: str, case_id: str,
             output_path: str) -> PDFAsset:

        self.validate(source)

        evidence_id = str(uuid.uuid4())
        source_path = Path(source)

        # Copy to case folder
        case_folder = Path(output_path) / case_id / "raw"
        case_folder.mkdir(parents=True, exist_ok=True)
        final_path = case_folder / f"{evidence_id}.pdf"
        shutil.copy2(source_path, final_path)

        # Extract page texts with page numbers preserved
        page_texts = self._extract_page_texts(str(final_path))

        # Detect document type from full text
        full_text = "\n\n".join(t for _, t in page_texts)
        doc_type = self._detect_doc_type(full_text)

        # Build sections — crosses page boundaries correctly
        sections = self._build_sections(page_texts)

        total_words = sum(s.word_count for s in sections)

        return PDFAsset(
            evidence_id=evidence_id,
            case_id=case_id,
            source_type=doc_type,
            original_name=source_path.name,
            stored_path=str(final_path),
            created_at=datetime.utcnow(),
            status="stored",
            total_pages=len(page_texts),
            total_words=total_words,
            sections=sections,
            metadata={
                "original_filename": source_path.name,
                "doc_type": doc_type,
                "total_pages": len(page_texts),
                "total_sections": len(sections),
                "total_words": total_words,
            },
        )

    # ── Internal helpers ─────────────────────────────────────────

    def _extract_page_texts(
        self, pdf_path: str
    ) -> List[Tuple[int, str]]:
        """
        Returns list of (page_number, cleaned_text).
        Tables are flattened to text so they are searchable.
        Page numbers are 1-indexed.
        """
        results = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = self._clean_text(text)

                # Flatten tables into searchable text
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        cells = [str(c).strip() for c in row if c]
                        text += "\n" + " | ".join(cells)

                if text.strip():
                    results.append((page_num, text.strip()))

        return results

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove form lines (------ or ====== lines)
        text = re.sub(r'^[-_=]{3,}\s*$', '', text,
                      flags=re.MULTILINE)
        return text.strip()

    def _detect_doc_type(self, text: str) -> str:
        sample = text[:3000].lower()
        for doc_type, patterns in _DOC_TYPE_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, sample, re.IGNORECASE):
                    return doc_type
        return "document"

    def _build_sections(
        self, page_texts: List[Tuple[int, str]]
    ) -> List[LegalSection]:
        """
        Core logic:
        1. Merge ALL page texts into a single stream of lines,
           tracking which page each line came from.
        2. Detect headings → use as section boundaries.
        3. Each section accumulates lines ACROSS page boundaries.
        4. Page span recorded as metadata only — never a boundary.
        """

        # Build line stream: [(page_number, line_text), ...]
        line_stream: List[Tuple[int, str]] = []
        for page_num, text in page_texts:
            for line in text.split("\n"):
                line_stream.append((page_num, line))

        # Walk the stream and split at headings
        raw_sections: List[Dict] = []
        current_title = "PREAMBLE"
        current_lines: List[str] = []
        current_pages: List[int] = []

        for page_num, line in line_stream:
            if self._is_heading(line):
                # Save current section before starting new one
                if current_lines:
                    raw_sections.append({
                        "title": current_title,
                        "lines": current_lines,
                        "pages": current_pages,
                    })
                current_title = line.strip()
                current_lines = []
                current_pages = [page_num]
            else:
                current_lines.append(line)
                if page_num not in current_pages:
                    current_pages.append(page_num)

        # Flush the final section
        if current_lines:
            raw_sections.append({
                "title": current_title,
                "lines": current_lines,
                "pages": current_pages,
            })

        # Build LegalSection objects
        sections: List[LegalSection] = []
        for raw in raw_sections:
            text = "\n".join(raw["lines"]).strip()
            if not text:
                continue

            pages = sorted(raw["pages"])
            section = LegalSection(
                section_id=str(uuid.uuid4()),
                title=raw["title"],
                text=text,
                page_span=PageSpan(
                    start_page=pages[0],
                    end_page=pages[-1],
                ),
                section_type=self._detect_section_type(raw["title"]),
                word_count=len(text.split()),
            )
            sections.append(section)

        return sections

    def _is_heading(self, line: str) -> bool:
        line = line.strip()
        if not line or len(line) < 3:
            return False
        for pattern, _ in _SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False

    def _detect_section_type(self, title: str) -> str:
        title_lower = title.lower().strip()
        for keyword, section_type in _SECTION_TYPE_MAP.items():
            if keyword in title_lower:
                return section_type
        return "document"
    
class ReaderRouter:
    @staticmethod
    def from_path(source):
        if source.startswith(("http://", "https://")):
            if "youtube.com" in source or "youtu.be" in source:
                return YTReader()
            raise ValueError(f"Unsupported URL: {source}")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Not found: {source}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return PDFReader()

        video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
        if ext in video_exts:
            return LocalReader()

        raise ValueError(f"Unsupported file type: {ext}")
        
    
    