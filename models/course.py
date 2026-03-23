from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class CourseCreate(BaseModel):
    title: str
    author: str = "Unknown"
    source: str = ""
    userId: str = "nicolas"
    exercisesTotal: int = 20
    allowDownload: Optional[bool] = True


class CourseUpdate(BaseModel):
    status: Optional[str] = None
    exercisesDone: Optional[int] = None
    allowDownload: Optional[bool] = None
    mcqStatus: Optional[str] = None
    mcqCount: Optional[int] = None


class Course(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    userId: str = "nicolas"
    title: str
    author: str = "Unknown"
    source: str = ""
    pdfUrl: Optional[str] = None
    status: str = "Not Started"
    exercisesTotal: int = 20
    exercisesDone: int = 0
    allowDownload: bool = True
    mcqStatus: str = "none"     # none | generating | ready | failed
    mcqCount: int = 0
    createdAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updatedAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class CourseResponse(BaseModel):
    id: str
    userId: str
    title: str
    author: str
    source: str
    pdfUrl: Optional[str]
    status: str
    exercisesTotal: int
    exercisesDone: int
    allowDownload: bool
    mcqStatus: str
    mcqCount: int
    createdAt: str
    updatedAt: str
