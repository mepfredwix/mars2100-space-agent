"""Main agent loop for the MARS 2100 Space Intelligence Agent."""

import logging
import sys
from typing import Any

from openai import OpenAI, OpenAIError

from config import ConfigurationError, create_client, load_settings
from tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

# This hard limit prevents an agent from running forever or generating
# unlimited API requests.
MAX_AGENT_ITERATIONS = 5

SYSTEM_PROMPT = """You are MARS 2100, a helpful Space Intelligence Agent.
Answer space-related questions accurately and concisely.
Use the available tools when their built-in information can help answer the
request. Read tool results before answering. Do not invent live mission status
or claim that the built-in data is real-time. If a tool returns an error, you
may correct the tool call or explain the limitation in your final response.
"""


class AgentError(RuntimeError):
    """A safe, user-facing agent error that contains no credentials."""


def _assistant_message(message: Any) -> dict[str, Any]:
    """Convert an SDK assistant message into a minimal history entry.

    Tool-call IDs and arguments must be copied exactly. They are matched with
    subsequent role="tool" messages so the model can read each observation.
    """
    history_message: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }

    if message.tool_calls:
        history_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]

    return history_message


def run_agent(user_goal: str, client: OpenAI, model: str) -> str:
    """Run the bounded reason -> act -> observe loop and return the answer."""
    if not user_goal.strip():
        raise AgentError("Please provide a non-empty goal.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal.strip()},
    ]

    logger.info("Starting agent with goal: %s", user_goal.strip())

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        logger.info(
            "=== Iteration %d/%d: requesting model decision ===",
            iteration,
            MAX_AGENT_ITERATIONS,
        )

        try:
            # Native tool calling returns structured calls; the model never
            # supplies Python code for us to evaluate.
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
        except OpenAIError as exc:
            # Log only the exception class, not the exception body/request,
            # which could contain sensitive connection or provider details.
            logger.error("Model request failed (%s).", type(exc).__name__)
            raise AgentError(
                "The model request failed. Check the .env values, network "
                "access, and whether the selected model supports tool calling."
            ) from exc

        if not response.choices:
            logger.error("The model returned no choices.")
            raise AgentError("The model returned no response choices.")

        choice = response.choices[0]
        message = choice.message
        logger.info("Model response received (finish_reason=%s).", choice.finish_reason)

        # Save the assistant turn before observations to preserve a valid
        # assistant(tool_calls) -> tool(tool_call_id) conversation sequence.
        messages.append(_assistant_message(message))
        tool_calls = message.tool_calls or []

        if not tool_calls:
            if message.content and message.content.strip():
                logger.info("Final response received; no tool call requested.")
                return message.content.strip()

            logger.error("The model returned neither content nor a tool call.")
            raise AgentError("The model returned an empty response.")

        logger.info("Model requested %d tool call(s).", len(tool_calls))

        # Execute every requested call, including calls made in parallel. Each
        # result gets its own tool message with the original provider call ID.
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments_json = tool_call.function.arguments
            logger.info(
                "Calling tool: name=%s, call_id=%s, arguments=%s",
                tool_name,
                tool_call.id,
                arguments_json,
            )

            # execute_tool uses a fixed allowlist and validates JSON arguments.
            # Validation failures are observations, allowing the model to retry.
            result_json = execute_tool(tool_name, arguments_json)
            logger.info(
                "Tool observation: name=%s, call_id=%s, result=%s",
                tool_name,
                tool_call.id,
                result_json,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_json,
                }
            )

    logger.warning("Stopped after reaching the %d-iteration limit.", MAX_AGENT_ITERATIONS)
    return (
        f"I reached the maximum of {MAX_AGENT_ITERATIONS} agent iterations "
        "before producing a final answer. Please try a narrower request."
    )


def main() -> int:
    """Load configuration, read a goal, run the agent, and print its answer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        settings = load_settings()
        user_goal = " ".join(sys.argv[1:]).strip() or input(
            "Enter a space-related goal: "
        ).strip()
        client = create_client(settings)
        answer = run_agent(user_goal, client, settings.model)
    except (ConfigurationError, AgentError) as exc:
        # These exceptions deliberately contain safe instructions, not secrets.
        logger.error("%s", exc)
        return 1
    except (KeyboardInterrupt, EOFError):
        logger.info("Agent run cancelled.")
        return 130
    except Exception as exc:  # Last-resort guard; never print raw SDK payloads.
        logger.error("Unexpected agent failure (%s).", type(exc).__name__)
        return 1

    print(f"\nFinal answer:\n{answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
