# Weld Quality Agent

A welding quality assistant built with Google's **Agent Development Kit (ADK)** and **Gemini 3.8 Flash**. A welding engineer can ask what the welding procedure specification (WPS) says, check actual current/voltage/travel speed against the WPS limits, get out-of-range flags from a sensor log, or upload a weld photo and have visible defects compared against the WPS acceptance criteria. Every WPS fact in an answer comes from a tool call (FAISS retrieval or a limits lookup), never from the model's memory, and the agent says so when the data is not on file.

## Architecture

```mermaid
flowchart LR
    U[User: question, CSV or weld photo] --> UI[ADK web UI]
    UI --> A[ADK Agent<br/>Gemini 3.8 Flash]
    A -- vision --> P[Weld photo<br/>read directly by Gemini]
    A -- tool call --> S[search_wps]
    A -- tool call --> C[check_weld_parameters]
    A -- tool call --> V[analyze_weld_csv]
    S --> F[(FAISS index<br/>all-MiniLM-L6-v2)]
    F --> W[weld_data.txt<br/>WPS document]
    C --> L[wps_limits.json]
    V --> L
    V --> D[Sensor CSV<br/>uploaded or sample]
```

The agent is one `Agent` object ([weld_agent/agent.py](weld_agent/agent.py)). Gemini decides which tool to call; ADK runs the Python function and sends the result back to Gemini, which writes the answer from that result.

## Tools

| Tool | What it does |
|---|---|
| `search_wps(query)` | Semantic search over the WPS document. One chunk per WPS section, embedded with `all-MiniLM-L6-v2`, stored in FAISS. Returns the top 3 sections with a source reference such as `weld_data.txt > PREHEAT AND INTERPASS`. |
| `check_weld_parameters(wps_id, weld_pass, current, voltage, travel_speed)` | Compares actual values with the min/max for that pass in `wps_limits.json` and returns PASS/FAIL per parameter. |
| `analyze_weld_csv(file_name, wps_id, weld_pass)` | Mean/min/max/std of current and voltage from a sensor log, plus the count and timestamps of samples outside the WPS range. Works on an uploaded CSV or the built-in `sample_weld_log.csv`. |
| Weld photo (no tool) | Gemini is multimodal, so it reads an uploaded photo directly, describes visible defects, then calls `search_wps` for the acceptance criteria. |

The WPS, the limits and the sample log are synthetic demo data.

## Example queries

Real runs of the agent (answers lightly trimmed). The line under each question shows the tool call Gemini chose.

**1. "What is the minimum preheat temperature in WPS-2026-001, and which shielding gas is used?"**
`search_wps("WPS-2026-001 preheat temperature shielding gas")`

> * **Minimum Preheat Temperature:** **100°C** minimum *(Source: `weld_data.txt > PREHEAT AND INTERPASS`)*
> * **Shielding Gas:** **80% Argon + 20% CO₂** (flow rate: 18 L/min) *(Source: `weld_data.txt > FILLER METAL`)*

**2. "For WPS-2026-001 root pass I measured 230 A, 22 V and 15 cm/min. Is that acceptable?"**
`check_weld_parameters(wps_id="WPS-2026-001", weld_pass="root", current=230, voltage=22, travel_speed=15)`

> No, these parameters are **not acceptable** (Overall: **FAIL**).
>
> | Parameter | Actual | Allowed Range | Result |
> | :--- | :--- | :--- | :--- |
> | Current (A) | 230 | 160 – 200 | **FAIL** |
> | Voltage (V) | 22 | 20 – 24 | **PASS** |
> | Travel Speed (cm/min) | 15 | 12 – 18 | **PASS** |
>
> Current is 30 A above the maximum limit.

**3. "Analyze sample_weld_log.csv against WPS-2026-001 for the root pass."**
`analyze_weld_csv(file_name="sample_weld_log.csv", wps_id="WPS-2026-001", weld_pass="root")`

> | Parameter | Allowed Range | Mean | Min | Max | Out of Range | Timestamps (s) |
> | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
> | Current (A) | 160 – 200 | 182.14 | 166.90 | 215.60 | 6 / 120 (5.0%) | 20.0 – 22.5 |
> | Voltage (V) | 20 – 24 | 22.16 | 21.16 | 25.02 | 4 / 120 (3.3%) | 41.0 – 42.5 |

**When the answer is not in the WPS** ("What is the hydrogen bake-out temperature and time required by WPS-2026-001?"), the agent searches, finds nothing, and replies that a hydrogen bake-out is *not specified* in WPS-2026-001 instead of guessing a value.

## Run it locally

Needs Python 3.10+ and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/the-anand-sharma/Weld_Assistant_RAG_Based.git
cd Weld_Assistant_RAG_Based
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r weld_agent/requirements.txt pytest
```

Create a `.env` file in the project root (it is git-ignored):

```
GOOGLE_API_KEY=your_key_here
```

Then:

```bash
pytest          # tool tests, no API key needed
adk web         # open http://localhost:8000 and pick weld_agent
```

The free Gemini tier allows about 20 requests per day per model, and one question uses two or more. If you hit the limit, set `WELD_AGENT_MODEL` to another model (for example `gemini-3.7-flash`) before running `adk web`.

## Deployment

Not deployed: it runs locally. The agent folder is laid out for ADK's Cloud Run command (`adk deploy cloud_run --with_ui weld_agent`, with the API key in Secret Manager), which needs a Google Cloud billing account.

## Project layout

```
weld_agent/
  agent.py            # the ADK agent: model, instruction, tools
  tools.py            # the three tool functions
  requirements.txt
  data/               # WPS document, limits, sample sensor log
tests/test_tools.py   # pytest tests for the tools
app.py, gemini_vecc.py  # earlier LangChain LCEL / Streamlit versions
```

## About

Built by [Anand Sharma](https://github.com/the-anand-sharma) · [LinkedIn](https://www.linkedin.com/in/anand-sharma-573527133/)
