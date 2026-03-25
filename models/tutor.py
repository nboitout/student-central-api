from pydantic import BaseModel
from typing import Optional


class ChatMessage(BaseModel):
    role: str   # "ai" | "student"
    text: str


class TutorProbeRequest(BaseModel):
    courseId: str
    question: str
    options: list[str]
    correctIndex: int
    selectedIndex: int
    isCorrect: bool
    explanation: str
    language: str = "en"


class TutorReplyRequest(BaseModel):
    courseId: str
    question: str
    options: list[str]
    correctIndex: int
    selectedIndex: int
    isCorrect: bool
    explanation: str
    language: str = "en"
    history: list[ChatMessage]


class TutorResponse(BaseModel):
    message: str
