from pydantic import BaseModel
from typing import Optional


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
    facultyInsight: str  # Human-readable explanation for the faculty dashboard
    studentFeedback: str # Personalised feedback shown to the student
