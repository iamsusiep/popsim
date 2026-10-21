"""A self-hosted decisions endpoint: Jev-shaped requests answered by autotrust/JEV-9B on vLLM.

Request:  {"state": str | JSON, "questions": {id: question}}
          question = {"type": "noul", "instructions": str}
                   | {"type": "choice", "instructions": str, "criteria": {option_id: description | None}}
Response: {"model": str, "answers": {id: answer}}
          answer = {"type": "noul", "noul": P(true)}
                 | {"type": "choice", "choice": argmax_id, "probabilities": {option_id: p}}
          Each answer also carries the model's raw "logprobs" per option and its prompt's "usage".

This is the shape of the hosted typed-decision APIs, cut down to the two primitives this
project uses. JEV-9B only takes `score` on a fixed 0-5 scale, so it is
left out. JEV-9B is an open (Apache-2.0) distillation of Jev 1.13; the template, slot
layout, head bias and temperatures below all follow its model card.

    uv run modal run popsim/decider.py     # load the model, answer the model card's examples
"""

import json
import math

import modal

MODEL = "autotrust/JEV-9B"
MODEL_REVISION = "7ac669b8fb476ba926491e3d1c39f9330d68dbed"
LETTERS = "ABCDEFGHIJKLMNOP"  # JEV-9B's choice verbalizers: 2-16 options
GPU = "H100"
TIMEOUT_S = 60 * 60  # caps what one call can spend: an hour of one H100


def options_of(question: dict) -> tuple[list[str], list[str]]:
    """The option ids, and the option lines as the model sees them."""

    if question["type"] == "noul":
        return ["false", "true"], ["false", "true"]
    if question["type"] == "choice":
        ids = list(question["criteria"])
        if not 2 <= len(ids) <= len(LETTERS):
            raise ValueError(f"choice takes 2-{len(LETTERS)} options, got {len(ids)}")
        lines = [f"{LETTERS[i]}) {question['criteria'][k] or k}" for i, k in enumerate(ids)]
        return ids, lines
    raise ValueError(f"unsupported question type {question['type']!r}")


def render(state: object, question: dict) -> str:
    """JEV-9B's decision template, exactly as its model card gives it."""

    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    _, lines = options_of(question)
    return (
        f"[kind] {question['type']}\n[state] {state}\n[question] {question['instructions']}\n"
        "[options]\n" + "\n".join(lines) + "\n[decision]:"
    )


def distribution(logprobs: list[float], bias: list[float], temperature: float) -> list[float]:
    """The decision head's calibrated distribution from vLLM's restricted log-probabilities.

    The head bias and per-kind temperature are applied here; the log-softmax normaliser vLLM
    applied cancels out in the softmax."""

    z = [(lp + b) / temperature for lp, b in zip(logprobs, bias, strict=True)]
    top = max(z)
    e = [math.exp(x - top) for x in z]
    return [x / sum(e) for x in e]


def answer(question: dict, ids: list[str], probs: list[float]) -> dict:
    if question["type"] == "noul":
        return {"type": "noul", "noul": probs[1]}
    best = max(range(len(ids)), key=probs.__getitem__)
    return {"type": "choice", "choice": ids[best], "probabilities": dict(zip(ids, probs))}


app = modal.App("popsim-decider")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.31.0", "huggingface-hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("popsim")
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    scaledown_window=5 * 60,
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
class Decider:
    max_model_len: int = modal.parameter(default=8192)

    @modal.enter()
    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        path = snapshot_download(MODEL, revision=MODEL_REVISION)
        with open(f"{path}/adapter_vllm/decision_head.json") as f:
            head = json.load(f)
        with open(f"{path}/calibration.json") as f:
            self.temperature = json.load(f)["per_kind"]
        self.first_slot = {kind: head["slots"]["ranges"][kind][0] for kind in ("noul", "choice")}
        self.verbalizers, self.bias = head["verbalizer_ids"], head["bias"]
        self.llm = LLM(
            path,
            enable_lora=True,
            max_lora_rank=32,
            logprobs_mode="processed_logprobs",  # logprobs respect allowed_token_ids
            max_model_len=self.max_model_len,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        )
        self.lora = LoRARequest("jev-decision", 1, f"{path}/adapter_vllm")
        self.tokenizer = self.llm.get_tokenizer()

    @modal.method()
    def decide(self, requests: list[dict]) -> list[dict]:
        """Answer many requests in one vLLM batch.

        Prompts go in request order, so one state's questions sit next to each other and reuse
        its cached prefix."""

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        jobs, prompts, params = [], [], []
        for r, request in enumerate(requests):
            for qid, question in request["questions"].items():
                ids, _ = options_of(question)
                slot = self.first_slot[question["type"]]
                tokens = self.verbalizers[slot : slot + len(ids)]
                text = render(request["state"], question)
                jobs.append((r, qid, question, ids, slot, tokens))
                prompts.append(
                    TokensPrompt(prompt_token_ids=self.tokenizer.encode(text, add_special_tokens=False))
                )
                params.append(
                    SamplingParams(
                        max_tokens=1, temperature=1.0, allowed_token_ids=tokens, logprobs=len(tokens)
                    )
                )

        outputs = self.llm.generate(prompts, params, lora_request=self.lora, use_tqdm=False)
        responses: list[dict] = [
            {"model": MODEL, "answers": {}, "usage": {"input_tokens": 0, "cached_tokens": 0}} for _ in requests
        ]
        for (r, qid, question, ids, slot, tokens), output in zip(jobs, outputs, strict=True):
            top = output.outputs[0].logprobs[0]
            logprobs = [top[t].logprob if t in top else -1e9 for t in tokens]
            probs = distribution(logprobs, self.bias[slot : slot + len(ids)], self.temperature[question["type"]])
            usage = {"input_tokens": len(output.prompt_token_ids), "cached_tokens": output.num_cached_tokens or 0}
            responses[r]["answers"][qid] = answer(question, ids, probs) | {"logprobs": dict(zip(ids, logprobs)), "usage": usage}
            responses[r]["usage"]["input_tokens"] += len(output.prompt_token_ids)
            responses[r]["usage"]["cached_tokens"] += output.num_cached_tokens or 0
        return responses


CARD_EXAMPLES = [  # the model card's own examples, with the values it reports
    (
        {
            "state": "SKU AX-330 stock at 8% of safety level; supplier late twice this quarter.",
            "questions": {
                "supplier": {
                    "type": "choice",
                    "instructions": "Supplier response for this scenario.",
                    "criteria": {"issue_warning": None, "renegotiate": None, "dual_source": None, "maintain": None},
                }
            },
        },
        "issue_warning 0.33, renegotiate 0.14, dual_source 0.53, maintain 0.001",
    ),
    (
        {
            "state": "Customer says the parcel arrived damaged and wants their money back.",
            "questions": {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
        },
        "P(true) 0.991",
    ),
]


@app.local_entrypoint()
def smoke() -> None:
    responses = Decider().decide.remote([request for request, _ in CARD_EXAMPLES])
    for (_, expected), response in zip(CARD_EXAMPLES, responses, strict=True):
        print(json.dumps(response["answers"]), "| card:", expected)
