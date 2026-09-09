"""Japanese text normalisation for the Kokoro TTS engine.

Why this exists at all, and why it cannot be skipped:

sherpa-onnx's Kokoro frontend splits text on `[一-鿿]` and sends everything in that
range to `ConvertChineseToTokenIDs`, which reads `lexicon-zh.txt`. That function
never receives the requested language — only the *non*-Chinese branch does. Japanese
kanji sit inside that range (`今` U+4ECA, `私` U+79C1, `何` U+4F55), so Japanese
arrives at the model as **Mandarin-pronounced kanji plus espeak-ja kana**. Measured
on Orin 6: `今日私何` produces 30682 samples under `ja`, `en-us`, `es` and `fr`
alike — the language setting has no effect on kanji whatsoever, while kana-only text
changes length 4x between `ja` and `en-us`.

No sherpa-onnx release ships a Japanese lexicon, so the fix has to happen before the
text reaches the engine: convert every kanji to kana here, leaving nothing in the CJK
range, so the whole utterance takes the espeak-ja path.

Same role as plugins/thai_frontend.py, which exists because sherpa-onnx could not be
handed raw Thai either.

Two stages, and the second is not optional:

1. **Dates and counters**, by rule. Japanese counter readings are irregular and
   context-dependent, and a morphological analyser gives the isolated-morpheme
   reading instead: Janome renders `10月` as `ツキ` and `1日` as `ニチ`, where a
   Japanese speaker says `ジュウガツ` and `ツイタチ`. Those have to be rewritten
   before tokenisation, from an explicit table.
2. **Everything else**, via Janome (Apache-2.0, pure Python, verified importable on
   cp38/jp5.11 and cp310/jp6.1). pykakasi does the same job but is GPL-3.0;
   fugashi/cutlet declare `requires_python >= 3.9` and so cannot run on jp5.11.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# ── numbers ───────────────────────────────────────────────────────────────────

_DIGITS = ("", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ")

# Sino-Japanese numerals, which is what counters take. Deliberately not the native
# ひとつ/ふたつ series — those go with different counters and are not what a date or
# a measurement uses.
_TENS = "ジュウ"
_HUNDREDS = "ヒャク"
_THOUSANDS = "セン"

# Irregular readings caused by sound changes; writing them out is shorter than
# encoding the euphonic rules, and these are the only cases below 10000.
_HUNDRED_IRREGULAR = {3: "サンビャク", 6: "ロッピャク", 8: "ハッピャク"}
_THOUSAND_IRREGULAR = {3: "サンゼン", 8: "ハッセン"}


def _read_under_10000(n: int) -> str:
    """0-9999 as Sino-Japanese numerals in katakana."""
    if n == 0:
        return "ゼロ"
    out = []
    thousands, n = divmod(n, 1000)
    hundreds, n = divmod(n, 100)
    tens, ones = divmod(n, 10)
    if thousands:
        out.append(_THOUSAND_IRREGULAR.get(thousands)
                   or ((_DIGITS[thousands] if thousands > 1 else "") + _THOUSANDS))
    if hundreds:
        out.append(_HUNDRED_IRREGULAR.get(hundreds)
                   or ((_DIGITS[hundreds] if hundreds > 1 else "") + _HUNDREDS))
    if tens:
        out.append((_DIGITS[tens] if tens > 1 else "") + _TENS)
    if ones:
        out.append(_DIGITS[ones])
    return "".join(out)


def read_number(n: int) -> str:
    """Any non-negative integer as katakana, grouped in Japanese 万/億 units.

    Japanese groups by four digits, not three, so 20260000 is ニセンニヒャクロクジュウ
    マン and not "twenty million". Getting the grouping wrong is the classic mistake
    when porting a Western number reader.
    """
    if n < 0:
        return "マイナス" + read_number(-n)
    if n < 10000:
        return _read_under_10000(n)
    units = ["", "マン", "オク", "チョウ"]
    parts = []
    idx = 0
    while n and idx < len(units):
        n, group = n // 10000, n % 10000
        if group:
            parts.append(_read_under_10000(group) + units[idx])
        idx += 1
    return "".join(reversed(parts))


# ── dates and counters ────────────────────────────────────────────────────────

# 月: 4, 7 and 9 are irregular (シ / シチ / ク, never ヨン / ナナ / キュウ).
_MONTHS = {
    1: "イチガツ", 2: "ニガツ", 3: "サンガツ", 4: "シガツ", 5: "ゴガツ", 6: "ロクガツ",
    7: "シチガツ", 8: "ハチガツ", 9: "クガツ", 10: "ジュウガツ", 11: "ジュウイチガツ",
    12: "ジュウニガツ",
}

# 日: the first ten days, plus 14/20/24, use native readings. Every other day is
# just the number + ニチ — 18 and 28 are regular, so the table below is complete.
_DAYS = {
    1: "ツイタチ", 2: "フツカ", 3: "ミッカ", 4: "ヨッカ", 5: "イツカ", 6: "ムイカ",
    7: "ナノカ", 8: "ヨウカ", 9: "ココノカ", 10: "トオカ", 14: "ジュウヨッカ",
    20: "ハツカ", 24: "ニジュウヨッカ",
}


def _read_month(n: int) -> str:
    return _MONTHS.get(n, read_number(n) + "ガツ")


def _read_day(n: int) -> str:
    return _DAYS.get(n, read_number(n) + "ニチ")


# Ordered longest-first so 年月日 is consumed as a date before 年 alone matches.
_DATE_FULL = re.compile(r"(\d{1,4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_DATE_MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_YEAR = re.compile(r"(\d{1,4})\s*年")
_MONTH = re.compile(r"(\d{1,2})\s*月")
_DAY = re.compile(r"(\d{1,2})\s*日")
_TIME = re.compile(r"(\d{1,2})\s*時\s*(\d{1,2})\s*分")
_HOUR = re.compile(r"(\d{1,2})\s*時")
_MINUTE = re.compile(r"(\d{1,2})\s*分")

# 時 and 分 irregulars are driven by the **trailing** digit, not by the whole value,
# so these tables cover the ones place and the readers below apply them positionally.
# A flat lookup keyed on the whole number gets 14時 (ジュウヨジ, not ジュウヨンジ) and
# 30分 (サンジュップン, not サンジュウフン) wrong.
_HOUR_ONES = {4: "ヨジ", 7: "シチジ", 9: "クジ"}
_MINUTE_ONES = {1: "イップン", 3: "サンプン", 4: "ヨンプン", 6: "ロップン",
                8: "ハップン"}


def _read_hour(n: int) -> str:
    tens, ones = divmod(n, 10)
    head = read_number(tens * 10) if tens else ""
    if ones in _HOUR_ONES:
        return head + _HOUR_ONES[ones]
    if ones == 0:
        return read_number(n) + "ジ"
    return head + read_number(ones) + "ジ"


def _read_minute(n: int) -> str:
    tens, ones = divmod(n, 10)
    if ones == 0 and tens:
        # 10分 ジュップン, 20分 ニジュップン, 30分 サンジュップン — the ジュウ
        # contracts to ジュッ before プン.
        base = read_number(n)
        assert base.endswith("ジュウ"), base
        return base[:-1] + "ップン"
    head = read_number(tens * 10) if tens else ""
    if ones in _MINUTE_ONES:
        return head + _MINUTE_ONES[ones]
    return head + read_number(ones) + "フン"


def normalize_dates(text: str) -> str:
    """Rewrite dates, times and their counters into katakana.

    Runs before Janome because Janome would give `月` the isolated reading `ツキ`
    and `日` the reading `ニチ`, so `10月1日` came out `10ツキ1ニチ` where a speaker
    says `ジュウガツツイタチ`.
    """
    text = _DATE_FULL.sub(
        lambda m: (read_number(int(m.group(1))) + "ネン"
                   + _read_month(int(m.group(2))) + _read_day(int(m.group(3)))), text)
    text = _DATE_MD.sub(
        lambda m: _read_month(int(m.group(1))) + _read_day(int(m.group(2))), text)
    text = _TIME.sub(
        lambda m: _read_hour(int(m.group(1))) + _read_minute(int(m.group(2))), text)
    text = _YEAR.sub(lambda m: read_number(int(m.group(1))) + "ネン", text)
    text = _MONTH.sub(lambda m: _read_month(int(m.group(1))), text)
    text = _DAY.sub(lambda m: _read_day(int(m.group(1))), text)
    text = _HOUR.sub(lambda m: _read_hour(int(m.group(1))), text)
    text = _MINUTE.sub(lambda m: _read_minute(int(m.group(1))), text)
    return text


# Any remaining bare integer. Left until after the counters, so a date's digits are
# never read twice.
_BARE_NUMBER = re.compile(r"\d+")


def normalize_numbers(text: str) -> str:
    """Read any leftover digit run as a Japanese numeral."""
    return _BARE_NUMBER.sub(lambda m: read_number(int(m.group(0))), text)


# ── kanji ─────────────────────────────────────────────────────────────────────

_CJK = re.compile(r"[一-鿿]")


def has_kanji(text: str) -> bool:
    """True when anything would still reach sherpa-onnx's Chinese branch."""
    return bool(_CJK.search(text))


class JapaneseFrontend:
    """Turns Japanese text into kana so it takes the espeak-ja path.

    Holds the Janome tokenizer, which loads a ~180 MB dictionary on construction —
    so build once per adapter, never per utterance.
    """

    def __init__(self):
        self._tokenizer = None
        try:
            from janome.tokenizer import Tokenizer
        except ImportError as error:
            # Refuse rather than warn, unlike the Chinese normaliser. Without this
            # the kanji do not merely lose their numbers — they are pronounced in
            # Mandarin, so the card would come up `running` and speak the wrong
            # language. Same call the Thai adapter makes about pythainlp.
            raise RuntimeError(
                "the Japanese voice needs janome (installed when ENABLE_JA_TTS=1); "
                "without it, kanji reach sherpa-onnx's Chinese branch and are "
                f"pronounced in Mandarin: {error}"
            ) from error
        self._tokenizer = Tokenizer()
        log.info("[tts] Japanese frontend ready (janome)")

    def to_kana(self, text: str) -> str:
        """Kanji -> katakana, via Janome's per-token readings."""
        parts = []
        for token in self._tokenizer.tokenize(text):
            reading = token.reading
            if not reading or reading == "*":
                # Punctuation, Latin and anything out-of-dictionary keep their
                # surface form; only kanji actually need converting.
                reading = token.surface
            parts.append(reading)
        return "".join(parts)

    def normalize(self, text: str) -> str:
        """Full pipeline: counters, then leftover numbers, then kanji."""
        if not text:
            return text
        out = self.to_kana(normalize_numbers(normalize_dates(text)))
        if has_kanji(out):
            # Not fatal — partial kana is still better than all-Mandarin — but it
            # means those characters will be read in Chinese, so say so once.
            log.warning(
                "[tts] Japanese text still contains kanji after conversion (%s); "
                "those characters will be pronounced in Mandarin",
                "".join(sorted(set(_CJK.findall(out))))[:20])
        return out
