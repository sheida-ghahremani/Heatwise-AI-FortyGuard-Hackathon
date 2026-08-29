from __future__ import annotations

import os
from typing import Iterable

import requests

from .models import UserProfile
from .routing import RouteResult


SYSTEM_PROMPT = """You are HeatWise Assistant, a concise bilingual (English/Persian)
urban-heat route explainer. Answer only from the supplied user profile, weather,
and calculated route metrics. Explain tradeoffs clearly. Never diagnose, claim
medical certainty, or tell a user that a route is guaranteed safe. Explain that
UTCI is an environmental thermal-stress index and does not include an age
coefficient. For strong or greater UTCI heat stress, recommend reducing exposure
and following official local heat guidance. Match the user's language."""


def _context_text(
    routes: Iterable[RouteResult],
    profile: UserProfile,
    origin: str,
    destination: str,
    humidity_pct: float,
    wind_mps: float,
) -> str:
    lines = [
        f"Trip: {origin} to {destination}",
        f"Profile: age {profile.age_group.value}; activity {profile.activity.value}",
        f"Weather inputs: humidity {humidity_pct:.0f}%; wind {wind_mps:.1f} m/s",
    ]
    for route in routes:
        lines.append(
            f"{route.name}: {route.distance_m/1000:.2f} km; {route.duration_min:.1f} min; "
            f"mean temperature {route.average_temp_c:.1f} C; shade {route.shade_pct:.0f}%; "
            f"mean UTCI {route.average_utci_c:.1f} C; UTCI exposure load "
            f"{route.utci_exposure_load:.1f} C-min above 26 C; activity-sensitive PET "
            f"{route.average_pet_c:.1f} C and PET load {route.heat_score:.1f} C-min above 29 C."
        )
    return "\n".join(lines)


def _local_answer(question: str, routes: list[RouteResult], profile: UserProfile) -> str:
    q = question.lower()
    fastest = next(route for route in routes if route.name == "Fastest")
    coolest = next(route for route in routes if route.name == "Lowest Heat Risk")
    balanced = next(route for route in routes if route.name == "Balanced")
    persian = any("\u0600" <= char <= "\u06ff" for char in question)
    if persian:
        if any(word in q for word in ("چرا", "کم خطر", "خنک", "سایه")):
            return (
                f"مسیر کم‌خطر {coolest.duration_min:.1f} دقیقه طول می‌کشد و حدود "
                f"{coolest.shade_pct:.0f}٪ سایه دارد؛ مسیر سریع‌تر {fastest.duration_min:.1f} دقیقه و "
                f"{fastest.shade_pct:.0f}٪ سایه دارد. بنابراین پیشنهاد کم‌خطر، زمان و سایه را با هم مقایسه می‌کند."
            )
        if any(word in q for word in ("بهترین", "کدام", "پیشنهاد")):
            return (
                f"برای گروه سنی {profile.age_group.value} و فعالیت {profile.activity.value}، گزینه‌ی Balanced "
                f"یک انتخاب میانه است: {balanced.duration_min:.1f} دقیقه، {balanced.distance_m/1000:.2f} کیلومتر، "
                f"و طبقه‌بندی UTCI برابر با {balanced.risk}. سن در UTCI ضریب عددی ندارد و این توصیه پزشکی نیست."
            )
        return "می‌توانی درباره‌ی بهترین مسیر، دلیل تفاوت مسیرها، زمان سفر، دما، سایه یا ریسک نسبی سؤال کنی."
    if any(word in q for word in ("why", "cool", "shade", "risk")):
        return (
            f"The lowest-risk route takes {coolest.duration_min:.1f} minutes with about {coolest.shade_pct:.0f}% "
            f"modeled shade, versus {fastest.duration_min:.1f} minutes and {fastest.shade_pct:.0f}% on the fastest route. "
            f"Its mean UTCI is {coolest.average_utci_c:.1f} °C ({coolest.risk}) and its activity-sensitive "
            f"PET is {coolest.average_pet_c:.1f} °C ({coolest.pet_category}). UTCI is an environmental "
            "thermal-stress index, not medical advice or a guarantee of safety."
        )
    return (
        f"For this {profile.activity.value.lower()} profile, Balanced takes {balanced.duration_min:.1f} minutes over "
        f"{balanced.distance_m/1000:.2f} km with mean UTCI {balanced.average_utci_c:.1f} °C "
        f"({balanced.risk.lower()}). Ask why a route was selected."
    )


def answer_question(
    question: str,
    routes: list[RouteResult],
    profile: UserProfile,
    origin: str,
    destination: str,
    humidity_pct: float,
    wind_mps: float,
) -> tuple[str, str]:
    """Return ``(answer, mode)`` using Groq/OpenAI, then local rules."""
    context = _context_text(routes, profile, origin, destination, humidity_pct, wind_mps)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"CURRENT CALCULATED DATA\n{context}\n\nUSER QUESTION\n{question}",
                    },
                ],
                "max_tokens": 350,
                "temperature": 0.2,
            },
            timeout=45,
        )
        if response.ok:
            answer = response.json()["choices"][0]["message"]["content"].strip()
            if answer:
                return answer, "Groq · GPT-OSS 20B"
        if response.status_code not in (401, 403, 429):
            response.raise_for_status()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        suffix = " · Groq unavailable" if groq_key else ""
        return _local_answer(question, routes, profile), f"Local fallback{suffix}"

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            "instructions": SYSTEM_PROMPT,
            "input": f"CURRENT CALCULATED DATA\n{context}\n\nUSER QUESTION\n{question}",
            "max_output_tokens": 350,
        },
        timeout=45,
    )
    if response.status_code == 429:
        try:
            error_code = response.json().get("error", {}).get("code")
        except ValueError:
            error_code = None
        if error_code == "insufficient_quota":
            return _local_answer(question, routes, profile), "Local fallback · OpenAI billing inactive"
    response.raise_for_status()
    payload = response.json()
    texts = [
        item.get("text", "")
        for output in payload.get("output", [])
        for item in output.get("content", [])
        if item.get("type") == "output_text"
    ]
    answer = "\n".join(text for text in texts if text).strip()
    if not answer:
        raise RuntimeError("OpenAI returned no text output")
    return answer, "OpenAI"
