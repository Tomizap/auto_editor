import fontforge
import os
import glob

FONT_NAME = "EmojiInline"
OUTPUT = "assets/emoji_font/EmojiInline.ttf"
PNG_DIR = "assets/twemoji"

# PUA codepoint -> list of possible twemoji filenames (try in order)
# (the script will also auto-fallback via glob if needed)
PUA_TO_TWEMOJI = {
    0xE001: ["26a0-fe0f.png", "26a0.png"],  # ⚠️
    0xE002: ["274c.png"],                   # ❌
    0xE003: ["1f4a1.png"],                  # 💡
    0xE004: ["1f4b0.png"],                  # 💰
    0xE005: ["1f92b.png"],                  # 🤫
    0xE006: ["26a1-fe0f.png", "26a1.png"],  # ⚡
    0xE007: ["2705.png"],                   # ✅
    0xE008: ["1f525.png"],                  # 🔥
    0xE009: ["1f4bc.png"],                  # 💼
    0xE00A: ["1f4e7.png"],                  # 📧
}

def find_png(candidates):
    # 1) exact match
    for name in candidates:
        p = os.path.join(PNG_DIR, name)
        if os.path.exists(p):
            return p

    # 2) tolerant fallback (handles packs that use slightly different naming)
    # e.g. "26a0*.png" or "1f4a1*.png"
    for name in candidates:
        stem = name.replace(".png", "")
        hits = sorted(glob.glob(os.path.join(PNG_DIR, stem + "*.png")))
        if hits:
            return hits[0]

    return None

# ---- create font
font = fontforge.font()
font.fontname = FONT_NAME
font.familyname = FONT_NAME
font.fullname = FONT_NAME

# Metrics tuned for subtitles. Adjust later if needed.
font.em = 1000
font.ascent = 800
font.descent = 200

missing = []

for codepoint, candidates in sorted(PUA_TO_TWEMOJI.items()):
    path = find_png(candidates)
    if not path:
        missing.append((codepoint, candidates))
        continue

    glyph = font.createChar(codepoint)
    glyph.importImage(path)     # IMPORTANT for PNG bitmaps
    glyph.width = 1000

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
font.generate(OUTPUT)
font.close()

if missing:
    print("WARN: missing PNG(s) for some glyphs:")
    for cp, cands in missing:
        print("  U+%04X -> tried: %s" % (cp, ", ".join(cands)))
else:
    print("OK: all glyphs generated")

print("OK: wrote", OUTPUT)
