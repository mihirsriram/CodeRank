from typing import Dict
from langchain.schema.runnable import Runnable, RunnableLambda
from coderank_lc.core.prompts import CONCISE_FIXER, EXPLAINER, OPTIMIZER
import requests, os
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Hugging Face model endpoints (using one model for all three)
HF_MODELS = {
    "concise": "https://xiiukibz8hcuvjog.us-east-1.aws.endpoints.huggingface.cloud",  # Phi-3-mini-128k-instruct
    "explainer": "https://xiiukibz8hcuvjog.us-east-1.aws.endpoints.huggingface.cloud",
    "optimizer": "https://xiiukibz8hcuvjog.us-east-1.aws.endpoints.huggingface.cloud",
}

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()


def _mock(style: str) -> str:
    """Offline fallback outputs."""
    if style == "concise":
        return "# Mock: concise response"
    if style == "explainer":
        return "# Mock: explainer response"
    if style == "optimizer":
        return "# Mock: optimizer response"
    return "# Mock: generic response"


def call_hf(model_url: str, prompt: str) -> str:
    """Generic HF Inference API call."""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
    payload = {
        "inputs": f"{prompt}",
        "parameters": {
            "max_new_tokens": 900,  # Increased for longer code + explanation
            "temperature": 0.35,
            "return_full_text": False,
        },
    }
    try:
        print(f"\n🚀 Calling model → {model_url}")
        r = requests.post(model_url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # Handle Hugging Face response formats
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        elif isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        else:
            return str(data)
    except Exception as e:
        return f"# [Error calling {model_url}: {e}]"


def make_agent(style: str) -> Runnable:
    """Return a LangChain Runnable for the given style."""
    def _call(query: str) -> str:
        # Handle both dict and string model configs
        model_cfg = HF_MODELS.get(style, HF_MODELS["concise"])
        model_url = model_cfg if isinstance(model_cfg, str) else model_cfg.get("url")

        # Select appropriate prompt
        prompt = {
            "concise": CONCISE_FIXER,
            "explainer": EXPLAINER,
            "optimizer": OPTIMIZER,
        }.get(style, CONCISE_FIXER).format(query=query)

        if HF_API_TOKEN and model_url:
            result = call_hf(model_url, prompt)
            if result.startswith("# [Error"):
                print(result)
            return result
        else:
            print("⚠️ No valid token or model URL — using mock response.")
            return _mock(style)
    return RunnableLambda(_call)


def generate_all(query: str, styles=("concise", "explainer", "optimizer")) -> Dict[str, str]:
    """Generate responses from all configured agents."""
    agents = {s: make_agent(s) for s in styles}
    results = {}
    print("🔍 HF token loaded:", bool(HF_API_TOKEN))

    for i, s in enumerate(styles):
        print(f"\n🚀 Generating with agent '{s}'")
        try:
            results[f"Agent-{i+1}-{s}"] = agents[s].invoke(query)
        except Exception as e:
            results[f"Agent-{i+1}-{s}"] = f"# [Agent {s} failed: {e}]"
    return results
