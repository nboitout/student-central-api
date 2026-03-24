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


def _parse_json_response(raw_text: str) -> dict | list:
    """Strip markdown fences and extra text, then parse JSON from model output."""
    raw_text = raw_text.strip()

    # Strip markdown fences
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    # For arrays — extract everything between first [ and last ]
    if raw_text.startswith("["):
        last_bracket = raw_text.rfind("]")
        if last_bracket != -1:
            raw_text = raw_text[:last_bracket + 1]

    # For objects — extract everything between first { and last }
    elif raw_text.startswith("{"):
        last_brace = raw_text.rfind("}")
        if last_brace != -1:
            raw_text = raw_text[:last_brace + 1]

    return json.loads(raw_text.strip())


def _build_image_content_blocks(pdf_images: list[str], course_title: str) -> list[dict]:
    """Build the user content blocks: text intro + one image per page."""
    content = [
        {
            "type": "text",
            "text": (
                f"Course: '{course_title}'\n\n"
                f"Here are {len(pdf_images)} page(s) from the course document."
            )
        }
    ]
    for b64_image in pdf_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_image}",
                "detail": "high"
            }
        })
    return content


async def generate_mcq_bank(
    course_title: str,
    pdf_images: list[str],
    course_id: str,
    count: int = 10,
) -> list[MCQQuestion]:
    """
    Generate a bank of `count` distinct MCQ questions from the PDF pages.
    All questions are grounded strictly in the document content.
    Called once at upload time — questions are stored in Cosmos DB.
    """
    client = get_openai_client()
    deployment = get_deployment()

    json_instruction = (
        f"Respond ONLY with valid JSON — an array of exactly {count} question objects. "
        "No other text, no markdown, no code fences. Format:\n"
        f'[{{"question": "...", "options": ["A text", "B text", "C text", "D text"], '
        '"correctIndex": 0, "explanation": "..."}, ...]\n'
        "correctIndex is 0-based (0=A, 1=B, 2=C, 3=D).\n"
        "Each explanation must reference the specific content from the document."
    )

    system_prompt = (
        f"You are an expert academic assessment designer. "
        f"You will receive pages from a course document as images. "
        f"Generate exactly {count} DISTINCT high-quality multiple-choice questions that:\n"
        "- Are STRICTLY grounded in the provided document — including charts, diagrams, visuals\n"
        "- Can only be answered by someone who has read this specific document\n"
        "- Cover different sections and concepts across the document (not repetitive)\n"
        "- Test deep conceptual understanding at master's level\n"
        "- Each has exactly 4 options (A, B, C, D)\n"
        "- Each has one clearly correct answer supported by the document\n"
        "- Each has 3 plausible distractors targeting common misconceptions\n"
        "- Are varied in format: some conceptual, some applied, some analytical\n"
        + json_instruction
    )

    user_content = _build_image_content_blocks(pdf_images, course_title)
    user_content[0]["text"] += (
        f"\n\nGenerate exactly {count} distinct MCQs strictly grounded in this document."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=4000,  # more tokens needed for 10 questions
    )

    raw = _parse_json_response(response.choices[0].message.content)

    # raw is a list of question dicts
    questions = []
    for item in raw[:count]:  # safety cap
        options = [
            MCQOption(letter=["A", "B", "C", "D"][i], text=opt)
            for i, opt in enumerate(item["options"])
        ]
        questions.append(MCQQuestion(
            question=item["question"],
            options=options,
            correctIndex=item["correctIndex"],
            explanation=item["explanation"],
            courseId=course_id,
        ))

    return questions


async def generate_mcq(
    course_title: str,
    pdf_images: list[str],
    course_id: str,
) -> MCQQuestion:
    """
    Generate a single MCQ — used as fallback if bank is not ready.
    """
    bank = await generate_mcq_bank(
        course_title=course_title,
        pdf_images=pdf_images,
        course_id=course_id,
        count=1,
    )
    return bank[0]


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
    correct_option = options[correct_index] if correct_index < len(options) else "unknown"

    json_instruction = (
        "Respond ONLY with valid JSON in this exact format, no other text, "
        "no markdown, no code fences:\n"
        '{"signal": "Strong", "confidence": "High", '
        '"facultyInsight": "2-3 sentences for the faculty dashboard.", '
        '"studentFeedback": "2-3 sentences of personalised feedback for the student."}\n'
        'signal must be exactly one of: "Strong", "Fragile", "Partial misconception", "Low mastery"\n'
        'confidence must be exactly one of: "High", "Medium", "Low"'
    )

    system_prompt = (
        "You are an expert educational psychologist evaluating student reasoning quality.\n"
        "Classify the student's understanding into one of four categories:\n"
        "- Strong: correct answer AND explanation shows deep understanding of the concept\n"
        "- Fragile: correct answer BUT explanation is weak, vague, or suggests guessing\n"
        "- Partial misconception: wrong answer BUT explanation shows partial understanding worth building on\n"
        "- Low mastery: wrong answer AND explanation shows fundamental confusion\n"
        + json_instruction
    )

    opts_text = "\n".join([
        f"{['A','B','C','D'][i]}: {opt}"
        for i, opt in enumerate(options) if i < 4
    ])

    user_content = (
        f"Question: {question}\n\n"
        f"Options:\n{opts_text}\n\n"
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

    raw = _parse_json_response(response.choices[0].message.content)

    return ReasoningSignal(
        signal=raw["signal"],
        confidence=raw["confidence"],
        facultyInsight=raw["facultyInsight"],
        studentFeedback=raw["studentFeedback"],
    )
