from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid


class QuestionRecord(BaseModel):
    position: int
    mcqId: str
    question: str
    options: list[str]
    correctIndex: int
    pageNumber: Optional[int] = None

    # Filled when student answers
    selectedIndex: Optional[int] = None
    isCorrect: Optional[bool] = None
    durationSec: Optional[int] = None
    answeredAt: Optional[str] = None

    # Filled in tutoring mode
    studentExplanation: Optional[str] = None
    evaluationSignal: Optional[str] = None
    evaluationConfidence: Optional[str] = None
    facultyInsight: Optional[str] = None
    studentFeedback: Optional[str] = None


class ChatEntry(BaseModel):
    role: str            # "ai" | "student"
    text: str
    questionPosition: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SessionSummary(BaseModel):
    totalQuestions: int
    correctCount: int
    totalDurationSec: int
    signalBreakdown: dict


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    courseId: str
    userId: str = "nicolas"
    mode: str = "tutoring"       # "tutoring" | "assessment"
    language: str = "en"
    status: str = "started"      # started | answering | reviewing | chatting | completed
    startedAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completedAt: Optional[str] = None
    questions: list[QuestionRecord] = []
    chatHistory: list[ChatEntry] = []
    summary: Optional[SessionSummary] = None


# ── Request models ────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    courseId: str
    userId: str = "nicolas"
    mode: str = "tutoring"
    language: str = "en"


class SessionAnswerRequest(BaseModel):
    position: int
    selectedIndex: int
    durationSec: int = 0


class SessionExplanationRequest(BaseModel):
    position: int
    studentExplanation: str


class SessionChatRequest(BaseModel):
    role: str
    text: str
    questionPosition: Optional[int] = None
