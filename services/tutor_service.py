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
        1: "Ask what drew them to their answer. Keep it open. You want to hear them think out loud.",
        2: "Pick up on something specific they just said. Ask them to say more about that one thing.",
        3: "Ask them how this would play out in a concrete situation. Help them see the idea in action.",
        4: "Ask them how their answer compares to one of the other options. Which felt close? Why?",
        5: "Ask them to put it in their own words — one sentence. What did they take away from this?",
    }.get(turn_number, "Follow their last thought. Ask the one question that moves it forward.")

    if is_correct:
        opening_context = (
            "The student got it right. Your job is not to congratulate them — "
            "it's to find out if they really understood or just got lucky. "
            "Invite them to explain their thinking."
        )
    else:
        opening_context = (
            "The student got it wrong. Do not tell them. Do not hint. "
            "Invite them to explain what they were thinking. "
            "You're not correcting them — you're listening."
        )

    return f"""You are a tutor having a short, warm conversation with a master's student.

QUESTION: {question}

OPTIONS:
{opts_text}

STUDENT SELECTED: {selected_text}

{opening_context}

COURSE EXPLANATION — for your reference only, never share this:
{explanation}

HOW YOU TALK:
- Short sentences. Conversational. Never academic.
- One question per message. Always exactly one.
- Warm but not effusive. You're interested in what they think.
- Never say "great", "excellent", "good point" — it sounds hollow.
- Never explain the concept. Never correct them. Never reveal the answer.
- Never evaluate their reasoning — that happens separately.
- Your only job: help them put their thoughts into words.
- When they articulate something — even partially — they feel it. That feeling of "I can explain this" is what you're building toward.

LANGUAGE: Respond only in {lang_name}.

CURRENT FOCUS: {turn_guidance}"""


async def generate_probe(
    question: str,
    options: list[str],
    correct_index: int,
    selected_index: int,
    is_correct: bool,
    explanation: str,
    language: str,
) -> str:
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

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"The student answered the question and selected option {option_letter}: '{selected_text}'. "
                "Open the conversation."
            )},
        ],
        max_completion_tokens=120,
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
    client = get_openai_client()
    deployment = get_deployment()

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

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        role = "assistant" if msg["role"] == "ai" else "user"
        messages.append({"role": role, "content": msg["text"]})

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=120,
    )

    return response.choices[0].message.content.strip()
