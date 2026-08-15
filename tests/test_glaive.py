from robo_agency.data import glaive

CHAT = (
    "USER: what's the weather in Paris?\n"
    'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": {"city": "Paris"}} '
    "<|endoftext|>\n"
    'FUNCTION RESPONSE: {"temp": 18}\n'
    "ASSISTANT: It is 18 degrees in Paris. <|endoftext|>\n"
)
SYSTEM = "You have access to the following functions: get_weather"


def test_roles_parsed_in_order():
    messages = glaive.parse_chat(CHAT)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]


def test_function_response_is_tool_role():
    """FUNCTION RESPONSE приходит от среды, а не от модели."""
    messages = glaive.parse_chat(CHAT)
    assert messages[2]["content"] == '{"temp": 18}'


def test_end_token_stripped():
    messages = glaive.parse_chat(CHAT)
    assert all("<|endoftext|>" not in m["content"] for m in messages)


def test_functioncall_payload_preserved():
    """Ради этого всё и затевалось: вызов инструмента должен дойти до модели."""
    messages = glaive.parse_chat(CHAT)
    assert '"name": "get_weather"' in messages[1]["content"]


def test_system_prepended_as_first_message():
    result = list(glaive.convert([{"system": SYSTEM, "chat": CHAT}]))
    messages = result[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "get_weather" in messages[0]["content"]


def test_row_without_assistant_is_skipped():
    assert list(glaive.convert([{"system": "", "chat": "USER: привет\n"}])) == []


def test_empty_chat_skipped():
    assert list(glaive.convert([{"system": SYSTEM, "chat": ""}])) == []


def test_format_detection():
    assert glaive.matches([{"system": "a", "chat": "b"}])
    assert not glaive.matches([{"messages": []}])
    assert not glaive.matches([])


def test_conversion_produces_usable_example():
    """Прогон 001 дал здесь ноль примеров — этот тест падал бы тогда."""
    result = list(glaive.convert([{"system": SYSTEM, "chat": CHAT}]))
    assert len(result) == 1
    assert len(result[0]["messages"]) == 5
