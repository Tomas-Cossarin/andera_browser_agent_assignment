import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
MODEL = "claude-opus-4-7"


def call_claude(query: str) -> str:
    """Call the LLM with the given query."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": query}
        ]
    )

    return message.content[0].text


def call_claude_with_image(query: str, img_b64: bytes) -> str:
    """Call the LLM with the given query and image."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": query
                    }
                ],
            }
        ],
    )

    return message.content[0].text


def call_claude_with_image_and_tools(query: str, img_b64: bytes, tools: list[dict]) -> anthropic.types.Message:
    """Call the LLM with the given query, image, and tools."""
    return client.messages.create(
        model=MODEL,
        max_tokens=4096,
        tools=tools,
        tool_choice={"type": "any"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": query
                    }
                ],
            }
        ],
    )
