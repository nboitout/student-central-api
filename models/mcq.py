from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime


class MCQGenerateRequest(BaseModel):
    courseId: str
    pdfUrl: Optional[str] = None
    courseTitle: str = ""


class MCQOption(BaseModel):
    letter: str
    text: str


class MCQQuestion(BaseModel):
    question: str
    options: list[MCQOption]
    correctIndex: int
    explanation: str
    courseId: str


class StoredMCQ(BaseModel):
    """A single MCQ stored in the mcqs Cosmos DB container."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    courseId: str
    userId: str = "nicolas"
    question: str
    options: list[MCQOption]
    correctIndex: int
    explanation: str
    createdAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class MCQEvaluateRequest(BaseModel):
    courseId: str
    question: str
    options: list[str]
    correctIndex: int
    selectedIndex: int
    studentExplanation: Optional[str] = None


class ReasoningSignal(BaseModel):
    signal: str          # "Strong" | "Fragile" | "Partial misconception" | "Low mastery"
    confidence: str      # "High" | "Medium" | "Low"
    facultyInsight: str
    studentFeedback: str
