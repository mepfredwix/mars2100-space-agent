# MARS 2100 Space Intelligence Agent

A minimal Python agent loop that uses the OpenAI Python SDK with an
OpenAI-compatible ai& Inference API. The agent can decide whether to call a
fixed set of space tools, execute those tools, read their JSON observations,
and continue reasoning for at most **5 iterations**.

## Requirements

- Python 3.11 or newer
- An ai& Inference API key
- The provider's exact OpenAI-compatible base URL
- An available model ID that supports **native function/tool calling**

No API key, provider URL, or model ID is included in this repository.
The two initial tools use small built-in datasets; they do not fetch live data.

## 1. Create a virtual environment and install dependencies

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version  # Confirm Python 3.11 or newer
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.

## 2. Configure the API

Create your local environment file:

```bash
cp .env.example .env
```

Open `.env` in a text editor and fill in all three values:

```dotenv
AI_BASE_URL=<exact OpenAI-compatible base URL from your provider>
AI_API_KEY=<your real API key>
AI_MODEL=<exact tool-calling-capable model ID from your provider>
```

- Consult the ai& Inference API documentation/dashboard for the exact values;
  this project intentionally does not guess them.
- Follow the provider's documentation about whether the base URL ends in `/v1`.
- Do not use the placeholder file as an actual configuration and do not commit
  `.env`. It is already excluded by `.gitignore`.
- Keep the real key out of shell commands, source code, screenshots, and logs.
  Environment variables already present in your shell take precedence over
  values in `.env`.

## 3. Run the first test

With the virtual environment active and `.env` configured:

```bash
python agent.py "Give me basic Mars facts and summarize Apollo 11."
```

A model that follows the tool-call flow should use `get_mars_info` and
`get_space_mission_info`, print a log entry for each decision/call/observation,
and then print a final answer. The exact wording and tool-call sequence can
vary by model.

You can also run `python agent.py` and enter the goal interactively.

## How the bounded loop works

1. Send the system prompt, user goal, and native tool schemas to the model.
2. Save the assistant response, including all tool-call IDs and arguments.
3. For each requested tool, validate the tool name and JSON arguments, execute
   only the matching allowlisted Python function, and append its result with
   the matching `tool_call_id`.
4. Send the enlarged history back to the model for another decision.
5. Stop as soon as the model returns a final answer, or stop after 5 iterations.
   Tool/API errors receive safe handling; no model-generated Python is run.

For another useful test of the small mission dataset, try:

```bash
python agent.py "What was Voyager 1?"
```

If configuration is incomplete, the program reports only the missing variable
names. API errors are logged by exception type rather than by dumping requests,
headers, credentials, or provider response bodies.
