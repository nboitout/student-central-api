import os
import json
from openai import AzureOpenAI
from models.mcq import MCQQuestion, MCQOption, ReasoningSignal


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )


def get_deployment() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


async def generate_mcq(course_title: str, pdf_sas_url: str | None, course_id: str) -> MCQQuestion:
    """
    Generate one MCQ question from the course PDF (or title if no PDF).
    Returns a structured MCQQuestion with 4 options, correct index, and explanation.
    """
    client = get_openai_client()
    deployment = get_deployment()

    # Build the prompt — with or without PDF
    if pdf_sas_url:
        system_prompt = """You are an expert academic assessment designer.
You will receive a PDF document from a university course.
Your task is to generate ONE high-quality multiple-choice question that:
- Tests conceptual understanding, not just recall
- Has exactly 4 options (A, B, C, D)
- Has one clearly correct answer
- Has plausible distractors that target common misconceptions
- Is appropriate for a master's level student

Respond ONLY with valid JSON in this exact format, no other text:
{
  "question": "...",
  "options": ["option A text", "option B text", "option C text", "option D text"],
  "correctIndex": 0,
  "explanation": "Clear explanation of why the correct answer is right and why the others are wrong."
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Generate one MCQ for this course: '{course_title}'. Use the PDF content provided.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pdf_sas_url},
                    },
                ],
            },
        ]
    else:
        system_prompt = """You are an expert academic assessment designer.
Generate ONE high-quality multiple-choice question for the given course title.
The question should test conceptual understanding at master's level.

Respond ONLY with valid JSON in this exact format, no other text:
{
  "question": "...",
  "options": ["option A text", "option B text", "option C text", "option D text"],
  "correctIndex": 0,
  "explanation": "Clear explanation of why the correct answer is right and why the others are wrong."
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Generate one MCQ for this university course: '{course_title}'",
            },
        ]

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0.7,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content)

    options = [
        MCQOption(letter=["A", "B", "C", "D"][i], text=opt)
        for i, opt in enumerate(raw["options"])
    ]

    return MCQQuestion(
        question=raw["question"],
        options=options,
        correctIndex=raw["correctIndex"],
        explanation=raw["explanation"],
        courseId=course_id,
    )


async def evaluate_reasoning(
    question: str,
    options: list[str],
    correct_index: int,
    selected_index: int,
    student_explanation: str | None,
) -> ReasoningSignal:
    """
    Evaluate the quality of a student's reasoning.
    Returns a ReasoningSignal: Strong / Fragile / Partial misconception / Low mastery.
    """
    client = get_openai_client()
    deployment = get_deployment()

    is_correct = selected_index == correct_index
    selected_option = options[selected_index] if selected_index < len(options) else "unknown"
    correct_option = options[correct_index]

    system_prompt = """You are an expert educational psychologist evaluating student reasoning quality.
You will receive a multiple-choice question, the student's selected answer, and their written explanation.
Your job is to classify the quality of the student's understanding into one of four categories:

- "Strong": Student selected correctly AND explanation shows deep understanding of the underlying concept
- "Fragile": Student selected correctly BUT explanation is weak, vague, or suggests guessing/pattern matching
- "Partial misconception": Student selected incorrectly BUT explanation shows partial understanding worth building on
- "Low mastery": Student selected incorrectly AND explanation shows fundamental confusion or lack of understanding

Respond ONLY with valid JSON in this exact format, no other text:
{
  "signal": "Strong" | "Fragile" | "Partial misconception" | "Low mastery",
  "confidence": "High" | "Medium" | "Low",
  "facultyInsight": "2-3 sentences for the faculty dashboard describing what this student's response reveals about their understanding.",
  "studentFeedback": "2-3 sentences of personalised, constructive feedback for the student."
}"""

    user_content = f"""Question: {question}

Options:
A: {options[0] if len(options) > 0 else ""}
B: {options[1] if len(options) > 1 else ""}
C: {options[2] if len(options) > 2 else ""}
D: {options[3] if len(options) > 3 else ""}

Correct answer: {correct_option}
Student selected: {selected_option} ({"CORRECT" if is_correct else "INCORRECT"})

Student's explanation: {student_explanation or "No explanation provided."}

Evaluate the quality of this student's reasoning."""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=500,
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content)

    return ReasoningSignal(
        signal=raw["signal"],
        confidence=raw["confidence"],
        facultyInsight=raw["facultyInsight"],
        studentFeedback=raw["studentFeedback"],
    )
