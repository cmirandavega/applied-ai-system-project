import os
import json
import logging

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "recommender.log")

_configured = False
_logger = logging.getLogger("vibematch")


def _configure() -> None:
    global _configured
    if _configured:
        return

    # Derive the directory from the current LOG_FILE value so monkeypatching works
    log_dir = os.path.dirname(os.path.abspath(LOG_FILE))
    os.makedirs(log_dir, exist_ok=True)

    # Close and remove any stale handlers before adding a new one
    for handler in _logger.handlers[:]:
        handler.close()
        _logger.removeHandler(handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    _logger.addHandler(file_handler)
    _logger.setLevel(logging.INFO)

    _configured = True


def log_request(user_input: str, profile: dict, results: list) -> None:
    _configure()
    if results:
        top = f"{results[0][0]['title']} (score: {results[0][1]:.4f})"
    else:
        top = "no results"
    _logger.info(
        "Input: %r | Profile: %s | Top result: %s",
        user_input,
        json.dumps(profile),
        top,
    )


def log_error(user_input: str, exc: Exception) -> None:
    _configure()
    _logger.error(
        "Input: %r | Error: %s: %s",
        user_input,
        type(exc).__name__,
        exc,
    )


def _summarize_trace(trace: list) -> str:
    """Condense the orchestrator trace into a compact one-line tool sequence."""
    steps = []
    for entry in trace:
        etype = entry.get("type")
        if etype == "tool_call":
            steps.append(entry.get("tool", "?"))
        elif etype == "fallback":
            steps.append("fallback")
        elif etype == "cap_hit":
            steps.append("cap_hit")
        elif etype == "final_message":
            steps.append("final_message")
    return " → ".join(steps) if steps else "(no tool calls)"


def log_orchestrator_run(profile: dict, results: list, trace: list, api_calls: int) -> None:
    """Log an agent-orchestration run. Tagged [AGENT] to distinguish it from the
    fixed-pipeline log_request entries."""
    _configure()
    if results:
        top = f"{results[0][0]['title']} (score: {results[0][1]:.4f})"
    else:
        top = "no results"
    _logger.info(
        "[AGENT] Profile: %s | Groq calls: %d | Tools: %s | Top result: %s",
        json.dumps(profile),
        api_calls,
        _summarize_trace(trace),
        top,
    )


def log_orchestrator_warning(reason: str, api_calls: int, max_turns: int) -> None:
    """Log a WARNING when the orchestrator hits its Groq-call safety cap."""
    _configure()
    _logger.warning(
        "[AGENT] Safety cap hit: %s (Groq calls: %d, max_turns: %d)",
        reason,
        api_calls,
        max_turns,
    )
