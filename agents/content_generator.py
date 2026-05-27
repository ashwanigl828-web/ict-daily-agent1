"""
ICT Daily Agent — Content Generation Agent
Uses Groq API (primary) or Google Gemini AI (fallback) to generate
engaging ICT content in Hindi with SVG diagrams, quiz questions,
fun facts, and summaries.
"""

import json
import logging
import time
import os
from typing import Optional

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AI Provider Selection: Groq (primary) → Gemini (fallback)
# ---------------------------------------------------------------------------
_USE_GROQ = bool(os.getenv("GROQ_API_KEY", ""))
_USE_GEMINI = bool(config.GEMINI_API_KEY) and not _USE_GROQ

# Maximum retries for API calls
_MAX_RETRIES = 3
_RETRY_DELAY_BASE = 4  # seconds, exponential back-off base


def _init_groq_client():
    """Initialize Groq client."""
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        client = Groq(api_key=api_key)
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"🚀 AI Provider: Groq ({model})")
        return client, model
    except ImportError:
        logger.error("groq package not installed! Run: pip install groq")
        return None, None


def _init_gemini_model():
    """Initialize Gemini model."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.8,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        logger.info(f"🚀 AI Provider: Gemini ({config.GEMINI_MODEL})")
        return model
    except Exception as e:
        logger.error(f"Gemini initialization failed: {e}")
        return None


# Initialize the active provider
_groq_client = None
_groq_model = None
_gemini_model = None

if _USE_GROQ:
    _groq_client, _groq_model = _init_groq_client()
elif _USE_GEMINI:
    _gemini_model = _init_gemini_model()
else:
    logger.warning("⚠️ No AI provider configured! Set GROQ_API_KEY or GEMINI_API_KEY.")


def _build_prompt(
    topic_name: str,
    class_standard: str,
    part_number: int,
    total_parts: int,
    subtopics: list[str],
    improvement_hints: str,
) -> str:
    """Build the full prompt for content generation.

    Args:
        topic_name: Topic title in Hindi.
        class_standard: Class number string ("8"–"12").
        part_number: Which part of the topic this is (1-indexed).
        total_parts: Total number of parts for this topic.
        subtopics: List of subtopic strings.
        improvement_hints: AI-generated suggestions from past feedback.

    Returns:
        The prompt string.
    """
    difficulty = config.CLASS_DIFFICULTY.get(class_standard, config.CLASS_DIFFICULTY["8"])
    level = difficulty["level"]
    description = difficulty["description"]
    word_limit = difficulty["word_limit"]
    quiz_diff = difficulty["quiz_difficulty"]

    subtopic_str = "\n".join(f"  - {st}" for st in subtopics) if subtopics else "  (कोई विशेष उपविषय नहीं)"

    part_instruction = ""
    if total_parts > 1:
        part_instruction = f"""
यह टॉपिक {total_parts} भागों में बँटा है। यह भाग {part_number}/{total_parts} है।
- भाग {part_number} के लिए उचित उपविषय कवर करो।
- पिछले भागों का संक्षिप्त रिकैप दो (अगर part > 1)।
- "अगले भाग में..." जोड़ो (अगर अभी last part नहीं है)।"""

    hints_section = ""
    if improvement_hints.strip():
        hints_section = f"""

🔧 *सुधार सुझाव (पिछले feedback के आधार पर):*
{improvement_hints}

ऊपर दिए सुझावों को इस कंटेंट में apply करो।"""

    prompt = f"""तुम एक भारतीय स्कूल शिक्षक हो जो कक्षा {class_standard} के छात्रों को ICT (Information and Communication Technology) सिखाते हो।

📋 *विषय:* {topic_name}
📚 *कक्षा:* {class_standard}
📊 *कठिनाई स्तर:* {level}
📝 *विवरण:* {description}
📏 *शब्द सीमा:* {word_limit}
{part_instruction}

*उपविषय:*
{subtopic_str}
{hints_section}

तुम्हें नीचे दिए JSON format में कंटेंट generate करना है। **पूरा कंटेंट हिंदी (देवनागरी लिपि) में होना चाहिए।**

**ज़रूरी नियम:**
1. भाषा सरल हिंदी में हो — जैसे बच्चों को समझा रहे हो। अंग्रेज़ी technical terms को brackets में हिंदी अनुवाद के साथ रखो।
2. कहानी या उदाहरण से शुरू करो — जैसे "सोचो अगर तुम्हारे पास कोई computer न हो..."
3. भारतीय संदर्भ के उदाहरण दो — UPI, Aadhaar, IRCTC, Digital India, Paytm, Google Pay, आदि।
4. इमोजी का भरपूर उपयोग करो 🎯📱💡🖥️
5. हर section को रोचक और engaging बनाओ।

**SVG Diagrams के नियम:**
- 2-3 SVG diagrams generate करो (flowchart, block diagram, hierarchy chart, etc.)
- हर SVG का width="400" और height="300" हो
- SVG में Hindi labels (Devanagari) का उपयोग करो
- रंगीन design बनाओ — fills, strokes, rounded rectangles, arrows
- SVG code complete और valid होना चाहिए
- Font family "Noto Sans Devanagari, Arial Unicode MS, sans-serif" use करो
- SVG simple और clear हो — ज़्यादा complex मत बनाओ

**Quiz के नियम ({quiz_diff}):**
- 4 multiple-choice questions दो
- हर question में 4 options (A, B, C, D) हों
- correct_answer में सही option का letter (A/B/C/D) दो
- Class level के हिसाब से difficulty adjust करो

**JSON Schema (STRICTLY follow करो):**
{{
  "topic_title": "विषय का शीर्षक हिंदी में",
  "introduction": "एक रोचक paragraph (50-80 words) जो topic introduce करे। कहानी या सवाल से शुरू करो।",
  "main_content": "विस्तृत explanation ({word_limit}). Headings ## use करो. हर concept को उदाहरण से समझाओ. Bullet points use करो.",
  "diagrams": [
    "<svg width='400' height='300' xmlns='http://www.w3.org/2000/svg'>...</svg>",
    "<svg width='400' height='300' xmlns='http://www.w3.org/2000/svg'>...</svg>"
  ],
  "fun_facts": [
    "🌟 रोचक तथ्य 1 हिंदी में",
    "🌟 रोचक तथ्य 2 हिंदी में",
    "🌟 रोचक तथ्य 3 हिंदी में"
  ],
  "quiz_questions": [
    {{
      "question": "प्रश्न हिंदी में?",
      "options": ["A) विकल्प 1", "B) विकल्प 2", "C) विकल्प 3", "D) विकल्प 4"],
      "correct_answer": "A"
    }}
  ],
  "summary": [
    "✅ मुख्य बिंदु 1",
    "✅ मुख्य बिंदु 2",
    "✅ मुख्य बिंदु 3"
  ]
}}

**IMPORTANT:**
- ONLY return valid JSON. No markdown, no code fences, no extra text.
- "diagrams" must be a JSON array of SVG strings (2 or 3 elements).
- "fun_facts" must have 3-5 items.
- "quiz_questions" must have exactly 4 items.
- "summary" must have 3-4 items.
- All content MUST be in Hindi (Devanagari)."""

    return prompt


def _parse_response(raw_text: str) -> dict:
    """Parse and validate the JSON response from AI.

    Handles minor formatting issues like markdown code fences.

    Args:
        raw_text: The raw text from AI's response.

    Returns:
        Parsed and validated content dictionary.

    Raises:
        ValueError: If parsing fails or required keys are missing.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s\nRaw text (first 500 chars): %s", exc, text[:500])
        raise ValueError(f"Failed to parse AI response as JSON: {exc}") from exc

    # Validate required keys
    required_keys = [
        "topic_title", "introduction", "main_content",
        "diagrams", "fun_facts", "quiz_questions", "summary",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"AI response missing required keys: {missing}")

    # Ensure correct types and apply defaults
    if not isinstance(data["diagrams"], list):
        data["diagrams"] = []
    if not isinstance(data["fun_facts"], list):
        data["fun_facts"] = []
    if not isinstance(data["quiz_questions"], list):
        data["quiz_questions"] = []
    if not isinstance(data["summary"], list):
        data["summary"] = []

    return data


def _call_groq(prompt: str) -> str:
    """Call Groq API and return the raw text response."""
    if not _groq_client:
        raise RuntimeError("Groq client not initialized")
    
    chat_completion = _groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert Indian school teacher. Always respond with valid JSON only. No markdown, no code fences, no extra text."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=_groq_model,
        temperature=0.8,
        max_completion_tokens=8192,
        response_format={"type": "json_object"},
    )
    
    return chat_completion.choices[0].message.content


def _call_groq_with_model(prompt: str, model_name: str) -> str:
    """Call Groq API with a specific model and return the raw text response."""
    if not _groq_client:
        raise RuntimeError("Groq client not initialized")
    
    chat_completion = _groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert Indian school teacher. Always respond with valid JSON only. No markdown, no code fences, no extra text."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=0.8,
        max_completion_tokens=8192,
        response_format={"type": "json_object"},
    )
    
    return chat_completion.choices[0].message.content


def _call_gemini(prompt: str) -> str:
    """Call Gemini API and return the raw text response."""
    if not _gemini_model:
        raise RuntimeError("Gemini model not initialized")
    
    response = _gemini_model.generate_content(prompt)
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text


def generate_content(
    topic_name: str,
    class_standard: str,
    part_number: int = 1,
    total_parts: int = 1,
    subtopics: Optional[list[str]] = None,
    improvement_hints: str = "",
) -> dict:
    """Generate complete ICT lesson content using Groq or Gemini AI.

    Args:
        topic_name: The topic title in Hindi.
        class_standard: Class number ("8"–"12").
        part_number: Current part number (1-indexed).
        total_parts: Total parts for this topic.
        subtopics: List of subtopic strings.
        improvement_hints: AI feedback-based improvement suggestions.

    Returns:
        Dictionary with keys: topic_title, introduction, main_content,
        diagrams (list of SVG strings), fun_facts, quiz_questions, summary.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    if subtopics is None:
        subtopics = []

    prompt = _build_prompt(
        topic_name=topic_name,
        class_standard=class_standard,
        part_number=part_number,
        total_parts=total_parts,
        subtopics=subtopics,
        improvement_hints=improvement_hints,
    )

    # Build list of AI providers to try (with fallback models)
    providers = []
    
    if _USE_GROQ and _groq_client:
        # Primary Groq model
        providers.append(("groq", _groq_model))
        # Fallback Groq models (separate per-model quotas!)
        fallback_models = ["llama-3.1-8b-instant", "gemma2-9b-it", "llama-3.3-70b-versatile"]
        for fm in fallback_models:
            if fm != _groq_model:
                providers.append(("groq", fm))
    
    if config.GEMINI_API_KEY:
        providers.append(("gemini", config.GEMINI_MODEL))

    if not providers:
        raise RuntimeError("❌ No AI provider configured! Set GROQ_API_KEY or GEMINI_API_KEY.")

    last_error: Optional[Exception] = None

    for provider_type, model_name in providers:
        provider_label = f"{'Groq' if provider_type == 'groq' else 'Gemini'} ({model_name})"
        
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(
                    "🤖 [%s] Generating content for '%s' class %s (attempt %d/%d)",
                    provider_label, topic_name, class_standard, attempt, _MAX_RETRIES,
                )
                
                if provider_type == "groq":
                    raw_text = _call_groq_with_model(prompt, model_name)
                else:
                    raw_text = _call_gemini(prompt)
                
                content = _parse_response(raw_text)

                logger.info(
                    "✅ Content generated — title: '%s', diagrams: %d, quiz: %d",
                    content.get("topic_title", "?"),
                    len(content.get("diagrams", [])),
                    len(content.get("quiz_questions", [])),
                )
                return content

            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("Parse error on attempt %d: %s", attempt, exc)
                last_error = exc

            except Exception as exc:
                error_str = str(exc)
                logger.warning("API error on attempt %d: %s", attempt, exc)
                last_error = exc
                
                # If rate limit, skip to next model immediately
                if "429" in error_str or "rate_limit" in error_str:
                    logger.info("⚡ Rate limit hit — trying next model...")
                    break

            # Exponential back-off before retry
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAY_BASE * (2 ** (attempt - 1))
                logger.info("Retrying in %d seconds...", delay)
                time.sleep(delay)

    # All providers exhausted
    error_msg = f"Content generation failed after trying all providers. Last error: {last_error}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

