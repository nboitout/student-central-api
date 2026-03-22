import os
import json
from openai import AzureOpenAI
from models.mcq import MCQQuestion, MCQOption, ReasoningSignal


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def get_deployment() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")


async def generate_mcq(course_title: str, pdf_sas_url: str | None, course_id: str) -> MCQQuestion:
    """
    Generate one MCQ question from the course PDF (or title if no PDF).
    Returns a structured MCQQuestion with 4 options, correct index, and explanation.
    """
    client = get_openai_client()
    deployment = get_deployment()

    json_instruction = (
        "Respond ONLY with valid JSON in this exact format, no other text, "
        "no markdown, no code fences:\n"
        '{"question": "...", "options": ["option A", "option B", "option C", "option D"], '
        '"correctIndex": 0, "explanation": "..."}'
    )

    if pdf_sas_url:
        system_prompt = (
            "You are an expert academic assessment designer. "
            "You will receive a PDF document from a university course. "
            "Generate ONE high-quality multiple-choice question that tests conceptual "
            "understanding at master's level. It must have exactly 4 options (A-D), "
            "one clearly correct answer, and plausible distractors targeting common misconceptions. "
            + json_instruction
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Generate one MCQ for this course: '{course_title}'. Use the PDF content provided."},
                    {"type": "image_url", "image_url": {"url": pdf_sas_url}},
                ],
            },
        ]
    else:
        system_prompt = (
            "You are an expert academic assessment designer. "
            "Generate ONE high-quality multiple-choice question for the given course title "
            "at master's level. It must have exactly 4 options (A-D), one clearly correct answer, "
            "and plausible distractors targeting common misconceptions. "
            + json_instruction
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate one MCQ for this university course: '{course_title}'"},
        ]

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=800,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    raw = json.loads(raw_text)

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

    json_instruction = (
        "Respond ONLY with valid JSON in this exact format, no other text, "
        "no markdown, no code fences:\n"
        '{"signal": "Strong", "confidence": "High", '
        '"facultyInsight": "...", "studentFeedback": "..."}\n'
        'signal must be one of: "Strong", "Fragile", "Partial misconception", "Low mastery"\n'
        'confidence must be one of: "High", "Medium", "Low"'
    )

    system_prompt = (
        "You are an expert educational psychologist evaluating student reasoning quality. "
        "Classify the student's understanding into one of four categories:\n"
        "- Strong: correct answer AND explanation shows deep understanding\n"
        "- Fragile: correct answer BUT explanation is weak or suggests guessing\n"
        "- Partial misconception: wrong answer BUT explanation shows partial understanding\n"
        "- Low mastery: wrong answer AND explanation shows fundamental confusion\n"
        + json_instruction
    )

    user_content = (
        f"Question: {question}\n\n"
        f"Options:\nA: {options[0] if len(options) > 0 else ''}\n"
        f"B: {options[1] if len(options) > 1 else ''}\n"
        f"C: {options[2] if len(options) > 2 else ''}\n"
        f"D: {options[3] if len(options) > 3 else ''}\n\n"
        f"Correct answer: {correct_option}\n"
        f"Student selected: {selected_option} ({'CORRECT' if is_correct else 'INCORRECT'})\n\n"
        f"Student's explanation: {student_explanation or 'No explanation provided.'}\n\n"
        "Evaluate the quality of this student's reasoning."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=500,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    raw = json.loads(raw_text)

    return ReasoningSignal(
        signal=raw["signal"],
        confidence=raw["confidence"],
        facultyInsight=raw["facultyInsight"],
        studentFeedback=raw["studentFeedback"],
    )
