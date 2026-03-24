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
    """Strip markdown fences and extra trailing text, then parse JSON."""
    raw_text = raw_text.strip()

    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    # Extract clean array
    if raw_text.startswith("["):
        last = raw_text.rfind("]")
        if last != -1:
            raw_text = raw_text[:last + 1]
    # Extract clean object
    elif raw_text.startswith("{"):
        last = raw_text.rfind("}")
        if last != -1:
            raw_text = raw_text[:last + 1]

    return json.loads(raw_text.strip())


def _build_image_content_blocks(pdf_images: list[str], course_title: str, count: int) -> list[dict]:
    content = [{
        "type": "text",
        "text": (
            f"Course: '{course_title}'\n\n"
            f"Here are {len(pdf_images)} page(s) from the course document.\n"
            f"Generate exactly {count} MCQ(s) grounded strictly in this content."
        )
    }]
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
    Generate a bank of `count` distinct MCQ questions from the PDF page images.
    Each question includes pageNumber — the 0-based index of the most relevant page.
    """
    client = get_openai_client()
    deployment = get_deployment()

    json_instruction = (
        f"Respond ONLY with a valid JSON array of exactly {count} objects. "
        "No other text, no markdown, no code fences. Format:\n"
        '[{"question": "...", "options": ["A text", "B text", "C text", "D text"], '
        '"correctIndex": 0, "explanation": "...", "pageNumber": 0}, ...]\n'
        "correctIndex is 0-based (0=A, 1=B, 2=C, 3=D).\n"
        f"pageNumber is the 0-based index of the page (0 to {len(pdf_images)-1}) "
        "most relevant to the question.\n"
        "Each explanation must reference specific content from the document."
    )

    system_prompt = (
        f"You are an expert academic assessment designer. "
        f"Generate exactly {count} DISTINCT high-quality MCQs from the course document pages.\n"
        "Requirements:\n"
        "- STRICTLY grounded in the document — including charts, diagrams, visuals\n"
        "- Cover different sections and concepts (not repetitive)\n"
        "- Master's level conceptual understanding\n"
        "- Each has 4 options (A-D), one correct answer, 3 plausible distractors\n"
        "- Varied: conceptual, applied, analytical\n"
        "- pageNumber must match the page where the question's content appears\n"
        + json_instruction
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _build_image_content_blocks(pdf_images, course_title, count)},
        ],
        max_completion_tokens=4000,
    )

    raw = _parse_json_response(response.choices[0].message.content)

    questions = []
    for item in raw[:count]:
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
            pageNumber=item.get("pageNumber", 0),
        ))

    return questions


async def evaluate_reasoning(
    question: str,
    options: list[str],
    correct_index: int,
    selected_index: int,
    student_explanation: str | None,
) -> ReasoningSignal:
    client = get_openai_client()
    deployment = get_deployment()

    is_correct = selected_index == correct_index
    selected_option = options[selected_index] if selected_index < len(options) else "unknown"
    correct_option = options[correct_index] if correct_index < len(options) else "unknown"

    json_instruction = (
        "Respond ONLY with valid JSON, no markdown, no code fences:\n"
        '{"signal": "Strong", "confidence": "High", '
        '"facultyInsight": "...", "studentFeedback": "..."}\n'
        'signal: "Strong" | "Fragile" | "Partial misconception" | "Low mastery"\n'
        'confidence: "High" | "Medium" | "Low"'
    )

    system_prompt = (
        "You are an expert educational psychologist evaluating student reasoning.\n"
        "- Strong: correct + deep understanding\n"
        "- Fragile: correct + weak/vague explanation\n"
        "- Partial misconception: wrong + shows partial understanding\n"
        "- Low mastery: wrong + fundamental confusion\n"
        + json_instruction
    )

    opts_text = "\n".join([f"{['A','B','C','D'][i]}: {opt}" for i, opt in enumerate(options) if i < 4])

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Question: {question}\n\nOptions:\n{opts_text}\n\n"
                f"Correct: {correct_option}\n"
                f"Selected: {selected_option} ({'CORRECT' if is_correct else 'INCORRECT'})\n\n"
                f"Explanation: {student_explanation or 'None provided.'}"
            )},
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
