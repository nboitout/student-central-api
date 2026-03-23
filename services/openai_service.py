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


async def generate_mcq(
    course_title: str,
    pdf_images: list[str],  # list of base64-encoded PNG strings
    course_id: str,
) -> MCQQuestion:
    """
    Generate one MCQ question grounded strictly in the PDF page images.
    GPT-5.2-chat sees the actual pages — including charts, diagrams, and visuals.
    """
    client = get_openai_client()
    deployment = get_deployment()

    json_instruction = (
        "Respond ONLY with valid JSON in this exact format, no other text, "
        "no markdown, no code fences:\n"
        '{"question": "...", "options": ["option A text", "option B text", "option C text", "option D text"], '
        '"correctIndex": 0, "explanation": "..."}\n'
        "correctIndex is 0-based (0=A, 1=B, 2=C, 3=D).\n"
        "The explanation must reference the specific content, figure, or concept from the document."
    )

    system_prompt = (
        "You are an expert academic assessment designer. "
        "You will receive pages from a course document as images. "
        "Your task is to generate ONE high-quality multiple-choice question that:\n"
        "- Is STRICTLY grounded in the provided document pages — including any charts, diagrams, or visuals\n"
        "- Can only be answered correctly by someone who has read this specific document\n"
        "- Tests deep conceptual understanding, not surface-level recall\n"
        "- Has exactly 4 options (A, B, C, D)\n"
        "- Has one clearly correct answer supported by the document\n"
        "- Has 3 plausible distractors targeting common misconceptions about this specific content\n"
        "- Is appropriate for a master's level student\n"
        "- If the document contains visuals (charts, graphs, diagrams), consider testing understanding of those too\n"
        + json_instruction
    )

    # Build content blocks: text instruction + one image per page
    user_content = [
        {
            "type": "text",
            "text": f"Course: '{course_title}'\n\nHere are {len(pdf_images)} page(s) from the course document. Generate one MCQ strictly grounded in this content."
        }
    ]

    for i, b64_image in enumerate(pdf_images):
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_image}",
                "detail": "high"
            }
        })

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=800,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps output
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
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
        for i, opt in enumerate(options)
        if i < 4
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

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps output
    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
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
