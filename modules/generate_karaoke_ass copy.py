# -*- coding: utf-8 -*-

from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# "mono"  = emojis Unicode classiques (STABLE)
# "color" = emojis bitmap inline via font custom (EXPÉRIMENTAL)
EMOJI_MODE = "mono"

UNICODE_EMOJIS = {

    # ======================
    # ⚠️ WARNING / ATTENTION
    # ======================
    "important": "⚠",
    "attention": "⚠",
    "alerte": "⚠",
    "danger": "⚠",
    "risque": "⚠",
    "grave": "⚠",
    "critique": "⚠",
    "urgent": "⚠",
    "warning": "⚠",
    "prudence": "⚠",

    # ======================
    # ❌ ERROR / PROBLEM
    # ======================
    "erreur": "✖",
    "problème": "✖",
    "probleme": "✖",
    "bug": "✖",
    "fail": "✖",
    "échec": "✖",
    "echec": "✖",
    "faux": "✖",
    "mauvais": "✖",
    "bloqué": "✖",
    "bloque": "✖",
    "cassé": "✖",
    "casse": "✖",

    # ======================
    # 💡 IDEA / TIP
    # ======================
    "astuce": "✦",
    "conseil": "✦",
    "idée": "✦",
    "idee": "✦",
    "tips": "✦",
    "hack": "✦",
    "solution": "✦",
    "stratégie": "✦",
    "strategie": "✦",
    "méthode": "✦",
    "methode": "✦",
    "approche": "✦",

    # ======================
    # 💰 MONEY / VALUE
    # ======================
    "argent": "$",
    "money": "$",
    "euro": "€",
    "euros": "€",
    "revenu": "$",
    "revenus": "$",
    "gagner": "$",
    "gagne": "$",
    "profit": "$",
    "profits": "$",
    "rentable": "$",
    "salaire": "$",
    "payer": "$",
    "paiement": "$",
    "cash": "$",

    # ======================
    # ⚡ SPEED / ACTION
    # ======================
    "rapide": "⚡",
    "vite": "⚡",
    "instant": "⚡",
    "instantané": "⚡",
    "instantane": "⚡",
    "direct": "➜",
    "immédiat": "⚡",
    "immediat": "⚡",
    "express": "⚡",
    "accélérer": "➚",
    "accelerer": "➚",

    # ======================
    # ✅ SIMPLE / VALIDATION
    # ======================
    "simple": "✔",
    "facile": "✔",
    "ok": "✔",
    "valide": "✔",
    "validé": "✔",
    "valider": "✔",
    "correct": "✔",
    "juste": "✔",
    "bon": "✔",
    "réussi": "✔",
    "reussi": "✔",

    # ======================
    # 🔥 POWER / PERFORMANCE
    # ======================
    "efficace": "▲",
    "efficacité": "▲",
    "efficacite": "▲",
    "puissant": "▲",
    "fort": "▲",
    "top": "▲",
    "meilleur": "▲",
    "performant": "▲",
    "performance": "▲",
    "optimisé": "▲",
    "optimise": "▲",

    # ======================
    # 🤖 AI / TECH
    # ======================
    "ia": "◆",
    "ai": "◆",
    "automatisation": "◆",
    "automation": "◆",
    "agent": "◆",
    "bot": "◆",
    "script": "◆",
    "code": "</>",
    "coder": "</>",
    "dev": "</>",
    "api": "⎔",
    "saas": "☁",
    "logiciel": "☁",

    # ======================
    # 📈 BUSINESS / GROWTH
    # ======================
    "business": "↗",
    "croissance": "↗",
    "scaler": "↗",
    "scale": "↗",
    "marketing": "📢",
    "vente": "🛒",
    "vendre": "🛒",
    "client": "◉",
    "clients": "◉",
    "prospect": "◎",
    "leads": "◎",
    "conversion": "◎",
    "trafic": "➚",

    # ======================
    # 🕒 TIME / EFFORT
    # ======================
    "temps": "⏱",
    "lent": "▽",
    "galère": "!",
    "fatiguant": "!",
    "chiant": "!",
    "complexe": "⧉",
    "difficile": "⧉",

    # ======================
    # 🧲 ENGAGEMENT / CTA
    # ======================
    "regarde": "►",
    "regardez": "►",
    "attention": "►",
    "écoute": "◄",
    "écoutez": "◄",
    "partage": "⟳",
    "like": "♥",
    "likes": "♥",
    "abonne": "✚",
    "abonne-toi": "✚",
}

DEFAULT_FONT = "Noto Sans"
EMOJI_FONT = "Noto Sans"


# ============================================================
# EMOJI PROVIDER
# ============================================================

def pick_emoji(word: str) -> str:
    w = word.lower()
    for key, emoji in UNICODE_EMOJIS.items():
        if key in w:
            return emoji
    return ""


def format_emoji(emoji: str) -> str:
    """
    Inline emoji.
    - mono: unicode emoji directly
    - color: switch font only for the emoji glyph, then restore DEFAULT_FONT
    """
    if not emoji:
        return ""

    if EMOJI_MODE == "color":
        # IMPORTANT: \fn expects a FONT NAME, not a style name.
        return r"{\fn" + EMOJI_FONT + r"}" + emoji + r"{\fn" + DEFAULT_FONT + r"}"
    else:
        return emoji


# ============================================================
# Merge apostrophe tokens (c ' est → c'est, l' argent → l'argent)
# ============================================================

def merge_apostrophe_words(words):
    """
    Canonical merge for apostrophes:
    - normalize ’ → '
    - remove space tokens
    - merge: c ' est → c'est
    - merge: l' argent → l'argent
    - guarantees NO SPACE around apostrophe
    """
    merged = []
    i = 0

    # normalize + remove pure spaces
    cleaned = []
    for w in words:
        txt = w.get("word")
        if not txt or not txt.strip():
            continue
        cleaned.append({
            **w,
            "word": txt.replace("’", "'")
        })

    while i < len(cleaned):
        cur = cleaned[i]
        w = cur["word"]

        # case: standalone apostrophe
        if w == "'" and merged and i + 1 < len(cleaned):
            prev = merged[-1]
            nxt = cleaned[i + 1]

            prev["word"] = prev["word"] + "'" + nxt["word"]
            prev["end"] = nxt.get("end", prev["end"])
            i += 2
            continue

        # case: word ending with apostrophe
        if w.endswith("'") and i + 1 < len(cleaned):
            nxt = cleaned[i + 1]
            merged.append({
                **cur,
                "word": w[:-1] + "'" + nxt["word"],
                "end": nxt.get("end", cur["end"]),
            })
            i += 2
            continue

        merged.append(cur)
        i += 1

    return merged




# ============================================================
# MAIN ASS GENERATOR
# ============================================================

def generate_karaoke_ass_tiktok_punchy(
    aligned: dict,
    out_ass_path: str,
    resolution=(1080, 1920),
    window: int = 2,
):
    W, H = resolution
    out_ass_path = Path(out_ass_path)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{DEFAULT_FONT},86,&H00FFFFFF,&H0000FF00,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,6,0,2,60,60,200,1
Style: EmojiInline,{EMOJI_FONT},86,&H00FFFFFF,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,60,60,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def ts(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    lines = []

    for seg in aligned.get("segments", []):
        raw_words = [w for w in seg.get("words", []) if w.get("word")]
        words = merge_apostrophe_words(raw_words)

        for i, w in enumerate(words):
            start, end = w["start"], w["end"]
            if end <= start:
                continue

            lo = max(0, i - window)
            hi = min(len(words), i + window + 1)
            chunk = words[lo:hi]

            rendered = []
            dur_ms = int((end - start) * 1000)
            punch_ms = min(120, dur_ms)

            for j, cw in enumerate(chunk):
                txt = cw["word"]

                if lo + j == i:
                    # emoji UNIQUEMENT sur le mot actif
                    emoji = pick_emoji(txt) if EMOJI_MODE == "mono" else ""
                
                    rendered.append(
                        r"{"
                        r"\1c&HFFFFFF&"
                        r"\fscx100\fscy100"
                        + rf"\t(0,{punch_ms},\fscx118\fscy118)"
                        + r"}"
                        + txt
                        + (" " + format_emoji(emoji) if emoji else "")
                        + r"{\r}"
                    )
                else:
                    rendered.append(r"{\1c&H00FF00&}" + txt)


            lines.append(
                f"Dialogue: 0,{ts(start)},{ts(end)},Default,,0,0,0,,{' '.join(rendered)}"
            )

    out_ass_path.parent.mkdir(parents=True, exist_ok=True)
    out_ass_path.write_text(header + "\n".join(lines), encoding="utf-8")
