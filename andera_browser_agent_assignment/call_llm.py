from call_claude import call_claude, call_claude_with_image, call_claude_with_image_and_tools

import anthropic
from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(3))
def call_llm(query: str, img_b64: bytes | None=None) -> str:
    """Call the LLM with the given query and optional image, and return the response."""
    if img_b64 is None:
        return call_claude(query)

    return call_claude_with_image(query, img_b64)


@retry(stop=stop_after_attempt(3))
def call_llm_tool_call(
    goal: str, steps_taken: str, elements: str, img_b64: bytes, has_multiple_objectives: bool, prev_tool_name: str | None, at_bottom_of_page: bool,
) -> tuple[str, str, int | None, str | None]:
    """Call the LLM with the current state and a set of tools, and return the tool call information."""
    reasoning_prompt = "The model MUST explain its thinking and anything it sees in the 'reasoning' param."
    tools = []

    if has_multiple_objectives:
        tools.append({
            "name": "objective_completed",
            "description":f"Call this tool if you think you have info for any of the remaining 'Objectives'. {reasoning_prompt}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of your thinking."
                    },
                },
                "required": ["reasoning"]
            }
        })
    else:
        tools.append({
            "name": "finish",
            "description": f"Mark the task as done. {reasoning_prompt}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of your thinking."
                    },
                },
                "required": ["reasoning"]
            }
        })

    tools += [
        {
            "name": "click",
            "description": f"Click a visible interactive element on the page using its element_id. {reasoning_prompt}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of your thinking."
                    },
                    "element_id": {
                        "type": "integer",
                        "description": "ID of the element from the visible element list that should be clicked."
                    }
                },
                "required": ["reasoning", "element_id"]
            }
        },
        {
            "name": "type",
            "description": f"Type 'text' into an 'element_id'. {reasoning_prompt}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of your thinking."
                    },
                    "element_id": {
                        "type": "integer",
                        "description": "ID of the INPUT element from the visible element list that you want to enter text into."
                    },
                    "text": {
                        "type": "string",
                        "description": "The text you want to enter into the INPUT element."
                    },
                },
                "required": ["reasoning", "element_id", "text"]
            }
        },
    ]

    if not at_bottom_of_page:
        tools.append(        {
            "name": "scroll",
            "description": f"Scroll DOWN the page. {reasoning_prompt}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of your thinking."
                    },
                },
                "required": ["reasoning"]
            }
        })

    if prev_tool_name != "screenshot":
        tools.append({
            "name": "screenshot",
            "description": f"Take a screenshot if explicitly instructed. {reasoning_prompt}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Short explanation of your thinking."
                    },
                },
                "required": ["reasoning"]
            }
        })


    query = f"""Goal:
{goal}

CRITICAL RULE:
If you have the answer to ANY remaining 'Objective', you MUST immediately call objective_completed.

Visible Elements:
{elements}

{steps_taken}"""

    response = call_claude_with_image_and_tools(query, img_b64, tools)

    return extract_tool_use(response)


def extract_tool_use(response: anthropic.types.Message) -> tuple[str, str, int | None, str | None]:
    """Extract the tool call information from the LLM response."""
    tool_calls = [
        block for block in response.content
        if block.type == "tool_use"
    ]

    if not tool_calls:
        raise ValueError("No tool_use found")

    tool_call = tool_calls[0]
    tool_input = tool_call.input

    reasoning = tool_input.get("reasoning")
    element_id = tool_input.get("element_id")
    text = tool_input.get("text")

    return tool_call.name, reasoning, element_id, text
