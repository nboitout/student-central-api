import os
from openai import AzureOpenAI


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


def get_deployment() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")


# BCP-47 → human-readable language name for the system prompt
LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "ro": "Romanian", "da": "Danish", "sv": "Swedish", "fi": "Finnish",
    "el": "Greek", "uk": "Ukrainian", "ru": "Russian", "hu": "Hungarian",
    "hr": "Croatian", "cs": "Czech", "bg": "Bulgarian", "sr": "Serbian",
    "tr": "Turkish",
}


def _build_system_prompt(
    question: str,
    options: list[str],
    correct_index: int,
    selected_index: int,
    is_correct: bool,
    explanation: str,
    language: str,
    turn_number: int = 1,
) -> str:
    lang_name = LANGUAGE_NAMES.get(language, "English")
    selected_text = options[selected_index] if selected_index < len(options) else "unknown"
    opts_text = "\n".join([f"{['A','B','C','D'][i]}: {opt}" for i, opt in enumerate(options) if i < 4])

    turn_guidance = {
        1: "Surface the student's belief — ask what reasoning led them to their answer.",
        2: "Challenge one assumption in the student's last message — ask them to examine it.",
        3: "Ask the student to apply the concept to a different scenario or context.",
        4: "Ask the student to compare their chosen option with the other distractors.",
        5: "Ask the student to synthesise their understanding in one sentence.",
    }.get(turn_number, "Continue the Socratic dialogue, pushing one level deeper.")

    return f"""You are a Socratic AI tutor helping a master's student reflect on their understanding of a course concept.

QUESTION: {question}

OPTIONS:
{opts_text}

THE STUDENT SELECTED: {selected_text} ({'CORRECT' if is_correct else 'INCORRECT'})

COURSE EXPLANATION (for your reference only — do NOT share this with the student):
{explanation}

YOUR RULES — follow all of these without exception:
1. Respond ONLY in {lang_name}
2. Ask EXACTLY ONE question per reply — never two
3. NEVER reveal the correct answer — let the student arrive there themselves
4. NEVER evaluate the student's reasoning — that happens separately after the chat
5. NEVER lecture, explain, or confirm — only probe with questions
6. Stay grounded to this specific question and its options — do not wander
7. Be warm but intellectually rigorous — treat the student as a capable adult

CURRENT TURN GUIDANCE: {turn_guidance}"""


async def generate_probe(
    question: str,
    options: list[str],
    correct_index: int,
    selected_index: int,
    is_correct: bool,
    explanation: str,
    language: str,
) -> str:
    """
    Generate the opening Socratic probe — the tutor's first message.
    Surfaces the student's reasoning without revealing the correct answer.
    """
    client = get_openai_client()
    deployment = get_deployment()

    system_prompt = _build_system_prompt(
        question=question,
        options=options,
        correct_index=correct_index,
        selected_index=selected_index,
        is_correct=is_correct,
        explanation=explanation,
        language=language,
        turn_number=1,
    )

    selected_text = options[selected_index] if selected_index < len(options) else "unknown"
    option_letter = ["A", "B", "C", "D"][selected_index] if selected_index < 4 else "?"

    user_message = (
        f"The student just answered the question by selecting option {option_letter}: '{selected_text}'. "
        f"Generate your opening probe to begin the Socratic dialogue."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=200,
    )

    return response.choices[0].message.content.strip()


async def generate_reply(
    question: str,
    options: list[str],
    correct_index: int,
    selected_index: int,
    is_correct: bool,
    explanation: str,
    language: str,
    history: list[dict],
) -> str:
    """
    Generate the tutor's next Socratic reply based on the conversation history.
    """
    client = get_openai_client()
    deployment = get_deployment()

    # Turn number = number of AI messages already sent + 1
    turn_number = sum(1 for m in history if m["role"] == "ai") + 1

    system_prompt = _build_system_prompt(
        question=question,
        options=options,
        correct_index=correct_index,
        selected_index=selected_index,
        is_correct=is_correct,
        explanation=explanation,
        language=language,
        turn_number=min(turn_number, 5),
    )

    # Build conversation messages from history
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        role = "assistant" if msg["role"] == "ai" else "user"
        messages.append({"role": role, "content": msg["text"]})

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=200,
    )

    return response.choices[0].message.content.strip()
