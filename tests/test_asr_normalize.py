import pytest

from robo_agency.asr.normalize import characters, normalize, words


def test_case_and_punctuation_removed():
    assert normalize("Сәлем, әлем!") == "сәлем әлем"


def test_kazakh_letters_survive():
    """Специфические буквы трогать нельзя: без них метрика станет неверной."""
    text = "әғқңөұүhі"
    assert normalize(text) == "әғқңөұүhі"


def test_kazakh_i_not_confused_with_russian():
    """Казахская і (U+0456) и русская и — разные буквы, склеивать их нельзя."""
    assert normalize("кіру") != normalize("киру")


@pytest.mark.parametrize(
    "latin,cyrillic",
    [("сaлем", "салем"), ("aкпaрaт", "акпарат"), ("оpтa", "орта")],
)
def test_latin_homoglyphs_folded_to_cyrillic(latin, cyrillic):
    """Whisper иногда выдаёт латинские a/e/o/c/p вместо кириллических.

    Произнесено при этом верно, и без приведения это считалось бы ошибкой
    распознавания.
    """
    assert normalize(latin) == normalize(cyrillic)


def test_whitespace_collapsed():
    assert normalize("  екі   сөз \n\t") == "екі сөз"


def test_internal_hyphen_kept():
    """Дефис внутри слова в казахском несёт смысл."""
    assert normalize("қаза-қазан") == "қаза-қазан"


def test_edge_hyphen_becomes_separator():
    assert normalize("сөз - екі") == "сөз екі"


def test_unicode_forms_equal():
    """Одна буква в разных формах Unicode должна совпадать после NFC.

    «й» записывается либо одной кодовой точкой (U+0439), либо как «и» плюс
    краткая (U+0438 U+0306). На вид одинаково, по байтам — нет, и без NFC
    правильно распознанное слово засчиталось бы как ошибка.
    """
    composed = "\u0439"
    decomposed = "\u0438\u0306"
    assert composed != decomposed
    assert normalize(composed) == normalize(decomposed)


def test_different_kazakh_letters_stay_different():
    """ә (U+04D9) и ӓ (U+04D3) — разные буквы, а не формы одной."""
    assert normalize("\u04d9") != normalize("\u04d3")


def test_empty_input():
    assert normalize("") == ""
    assert words("") == []
    assert characters("") == []


def test_words_and_characters():
    assert words("Екі сөз") == ["екі", "сөз"]
    assert characters("Екі сөз") == list("екісөз")


def test_digits_preserved():
    assert normalize("2026 жыл") == "2026 жыл"
