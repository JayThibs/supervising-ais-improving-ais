import os
import re
from typing import Dict, List, Optional, Tuple


from auto_finetuning_helpers import make_api_request, load_api_key
from stats import benjamini_hochberg_correction



STEP1_INSTR = (
    "We are investigating the side effects of a particular intervention on a language model. "
    "We have a starting model (which we will call Model 1) and a modified version of that same model (called Model 2). "
    "We have generated an extensive set of natural language hypotheses that each describe a particular difference between these two models. "
    "We now wish to analyze these hypotheses.\n\n"
    "We will perform the following analyses:\n"
    "1: review the hypotheses to identify and highlight particularly interesting, relevant, or surprising side effects that merit closer attention.\n"
    "2: identify recurring themes or patterns in the discovered side effects, revealing systematic changes that might not be apparent from individual hypotheses alone.\n"
    "3: compile a comprehensive summary of all verified behavioral differences across the validated hypotheses, providing a holistic view of how the intervention has altered model behavior.\n"
    "Start by executing step 1 of the analysis: review the hypotheses to identify and highlight particularly interesting, relevant, or surprising side effects that merit closer attention. "
    "Aim to be concise. You're just highlighting a handful (let's say, 3 or 4 at most) of specific hypotheses that are particularly stand-out. "
    "Don't summarize multiple hypotheses, and avoid selecting redundant hypotheses. Format your response in LaTeX as a \\begin\{itemize\} list environment, with each item being a single hypothesis. Provide the full hypothesis text in the item, not a summary. Remember to use consistent LaTeX style formatting (\\textbf{}, `` as open quotes, etc)."
)

STEP2_INSTR = (
    "Now move on to step 2: identify recurring themes or patterns in the discovered side effects, "
    "revealing systematic changes that might not be apparent from individual hypotheses alone.\n"
    "Here, you're concisely summarizing the common effects that can be extracted by comparing multiple hypotheses. "
    "Revisit the hypotheses to identify common patterns among them. For each pattern you highlight, refer back to the hypotheses that support it.\n"
    "Organize your response as a nested list of \\begin\{itemize\} environments, with similar changes grouped together under a single top-level \\item. E.g.,\n"
    "\\begin\{itemize\}\n"
    "  \\item [Category 1] \\begin\{itemize\}\n"
    "    \\item [Specific change 1]\n"
    "    \\item ...\n"
    "    \\item [Specific change N]\n"
    "  \\end\{itemize\}\n"
    "  \\item [Category 2] \\begin\{itemize\}\n"
    "    \\item [Specific change 1]\n"
    "    \\item ...\n"
    "    \\item [Specific change N]\n"
    "  \\end\{itemize\}\n"
    "\\end\{itemize\}\n"
)

STEP3_INSTR = (
    "Finally, move on to step 3: compile a comprehensive summary of all verified behavioral differences across the validated hypotheses, "
    "providing a holistic view of how the intervention has altered model behavior. This approach aims to extract maximum insight from the outputs, "
    "moving beyond individual difference statements to understand broader patterns of behavioral change.\n"
    "I.e., if even a single hypothesis mentions a particular difference between the models, add it to the list. "
    "Go hypothesis by hypothesis and note any that add new distinguishing features. Then, compile all the differences between the models into another nested list of \\begin\{itemize\} environments, with one item per difference. "
    "You can think of this as an extended version of step 2, where the inclusion criteria are relaxed so that even if a single hypothesis mentions a particular difference between the models, it is added to the list. Keep in mind that you may need to create new categories to accommodate the expanded set of differences."
    "Again, highlight the supporting hypothesis / hypotheses.\n"
    "[This may take a while, so don't be afraid to think on it for an extended time.]"
)

def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_").lower()


def format_hypotheses_for_labeler(hypotheses: List[str]) -> str:
    """
    Returns the exact preamble + numbered hypotheses block specified in the prompt.
    """
    lines = ["Note: Model 1 is the base model. Model 2 is the intervention model.", ""]
    for i, h in enumerate(hypotheses, start=1):
        lines.append(f"Hypothesis {i}: {h}")
    return "\n".join(lines)

def _filter_indices_by_bh(pvals: List[float], alpha: float) -> List[int]:
    if not pvals:
        return []
    # Compute BH and map threshold to significant indices
    n_significant, thr, significant_mask = benjamini_hochberg_correction(pvals, alpha=alpha)
    return [i for i, p in enumerate(pvals) if p <= thr]

def select_hypotheses_for_summary(
    data: Dict[str, List],
    filter_type: str = "none",
    bh_alpha: float = 0.05,
    min_accuracy: Optional[float] = None,
    top_k: Optional[int] = None,
) -> List[str]:
    """
    Choose which hypotheses to include in the summarization phase.
    """
    n = len(data["hypotheses"])
    indices = list(range(n))
    # Accuracy gate
    if min_accuracy is not None:
        indices = [i for i in indices if i < len(data["accuracies"]) and data["accuracies"][i] is not None and data["accuracies"][i] >= min_accuracy]

    # P-value BH filter
    if filter_type != "none":
        pv_map = {
            "binomial": data.get("binomial_p_values", []),
            "permutation": data.get("permutation_p_values", []),
            "crossval": data.get("cross_val_permutation_test_p_values", []),
        }
        pvals = pv_map.get(filter_type, [])
        sig_idx = set(_filter_indices_by_bh(pvals, bh_alpha))
        indices = [i for i in indices if i in sig_idx]

    # Optional top-k by accuracy (descending)
    if top_k is not None and top_k > 0 and len(data["accuracies"]) == n:
        indices = sorted(indices, key=lambda i: data["accuracies"][i], reverse=True)[:top_k]

    return [data["hypotheses"][i] for i in indices]

def run_progressive_summary(
    name: str,
    hypotheses: List[str],
    api_provider: str,
    model_str: str,
    api_key_path: str,
    save_dir: str = "summaries",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    max_thinking_tokens: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[str, str, str]:
    """
    Execute the 3-step progressive summarization, saving artifacts to disk.
    """

    # Resolve API key
    resolved_key = load_api_key(api_key_path)

    # Build initial block with hypotheses
    hypotheses_block = format_hypotheses_for_labeler(hypotheses)

    # Build conversation & prompts
    base_context = (
        "We provide the Labeler API LLM with a list of validated hypotheses formatted as so:\n\n"
        "'''\n" + hypotheses_block + "\n'''"
    )

    # Step 1
    step1_prompt = base_context + "\n\n" + STEP1_INSTR
    print("\n[Progressive summary] Running step 1 (highlights)...")
    step1_resp = make_api_request(
        prompt=step1_prompt,
        api_provider=api_provider,
        model_str=model_str,
        api_key=resolved_key,
        max_tokens=max_tokens,
        max_thinking_tokens=max_thinking_tokens,
        temperature=temperature,
        request_info={"pipeline_stage": "progressive_step_1", "analysis_name": name},
        logging_level="DEBUG" if verbose else "INFO",
    )

    # Step 2
    step2_prompt = (
        step1_prompt
        + "\n\n--- Model response to step 1 ---\n"
        + step1_resp
        + "\n--- End response to step 1 ---\n\n"
        + STEP2_INSTR
    )
    print("[Progressive summary] Running step 2 (patterns)...")
    step2_resp = make_api_request(
        prompt=step2_prompt,
        api_provider=api_provider,
        model_str=model_str,
        api_key=resolved_key,
        max_tokens=max_tokens,
        max_thinking_tokens=max_thinking_tokens,
        temperature=temperature,
        request_info={"pipeline_stage": "progressive_step_2", "analysis_name": name},
        logging_level="DEBUG" if verbose else "INFO",
    )

    # Step 3
    step3_prompt = (
        step2_prompt
        + "\n\n--- Model response to step 2 ---\n"
        + "\n\n" + STEP3_INSTR
    )
    print("[Progressive summary] Running step 3 (comprehensive list)...")
    step3_resp = make_api_request(
        prompt=step3_prompt,
        api_provider=api_provider,
        model_str=model_str,
        api_key=resolved_key,
        max_tokens=max_tokens,
        max_thinking_tokens=max_thinking_tokens,
        temperature=temperature,
        request_info={"pipeline_stage": "progressive_step_3", "analysis_name": name},
        logging_level="DEBUG" if verbose else "INFO",
    )

    # Save artifacts
    os.makedirs(save_dir, exist_ok=True)
    base = _sanitize_filename(name or "current_analysis")
    step1_path = os.path.join(save_dir, f"{base}_step1_highlights.tex")
    step2_path = os.path.join(save_dir, f"{base}_step2_patterns.tex")
    step3_path = os.path.join(save_dir, f"{base}_step3_comprehensive.tex")
    convo_path = os.path.join(save_dir, f"{base}_conversation.txt")

    with open(step1_path, "w", encoding="utf-8") as f:
        f.write(step1_resp)
    with open(step2_path, "w", encoding="utf-8") as f:
        f.write(step2_resp)
    with open(step3_path, "w", encoding="utf-8") as f:
        f.write(step3_resp)

    with open(convo_path, "w", encoding="utf-8") as f:
        f.write("=== Hypotheses Block ===\n")
        f.write(hypotheses_block + "\n\n")
        f.write("=== Step 1 Prompt ===\n")
        f.write(step1_prompt + "\n\n")
        f.write("=== Step 1 Response ===\n")
        f.write(step1_resp + "\n\n")
        f.write("=== Step 2 Prompt ===\n")
        f.write(step2_prompt + "\n\n")
        f.write("=== Step 2 Response ===\n")
        f.write(step2_resp + "\n\n")
        f.write("=== Step 3 Prompt ===\n")
        f.write(step3_prompt + "\n\n")
        f.write("=== Step 3 Response ===\n")
        f.write(step3_resp + "\n")

    print(f"[Progressive summary] Saved:\n  {step1_path}\n  {step2_path}\n  {step3_path}\n  {convo_path}")
    return step1_resp, step2_resp, step3_resp



def score_hypotheis_diversity(
    hypotheses: List[str],
    api_provider: str,
    model_str: str,
    api_key_path: str,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    max_thinking_tokens: Optional[int] = 20000,
    verbose: bool = False,
) -> int:
    """
    Score the diversity of the hypotheses. Basically, we just ask the LLM assistant to count how
    many unique differences the provided hypotheses collectively describe. The idea is that many
    of the hypotheses will be redundant, constantly repeating the same discovered differences
    between the models.

    Args:
        hypotheses (List[str]): The list of hypotheses to score. Each describes a collection of differences
        between two the models under investigation, namely Model 1 and Model 2.
        api_provider (str): The API provider to use for the API call.
        model_str (str): The model to use for the API call.
        api_key_path (str): The path to the API key file.
        max_tokens (int): The maximum number of tokens to generate (both output and COT).
        temperature (float): The temperature to use for the API call.
        max_thinking_tokens (Optional[int]): The maximum number of thinking tokens to generate.
        verbose (bool): Whether to print verbose logging of API calls and model interactions.
    Returns:
        int: The number of unique differences described by the hypotheses.
    """
    if not hypotheses:
        return 0

    resolved_key = load_api_key(api_key_path)
    hypotheses_block = format_hypotheses_for_labeler(hypotheses)

    prompt = (
        "You are analyzing behavioral differences between two language models to determine the side "
        "effects of an intervention on the language model.\n"
        "Model 1 is the base model. Model 2 is the intervention model.\n\n"
        "You are given a list of hypotheses. Each hypothesis describes one or more differences "
        "between Model 1 and Model 2. Different hypotheses may describe the same underlying "
        "difference using different wording.\n\n"
        "Your task:\n"
        "1. Infer how many *unique* underlying differences are described across the ENTIRE list.\n"
        "   - If multiple hypotheses clearly describe the same behavioral change, they count as ONE difference.\n"
        "   - If a single hypothesis describes multiple independent differences, each counts separately.\n"
        "2. Ignore how many times each difference appears; only count distinct differences.\n\n"
        "Here are the hypotheses:\n"
        "'''\n"
        f"{hypotheses_block}\n"
        "'''\n\n"
        "At the very end of your response, output the total count of unique differences as a single integer on a new line. "
        "Format the final line exactly as: 'COUNT: <integer>'\n\n"
    )

    # print(f"Prompt: {prompt}")
    # print(f"API provider: {api_provider}")
    # print(f"Model: {model_str}")
    # print(f"API key: {api_key_path}")
    # print(f"Max tokens: {max_tokens}")
    # print(f"Max thinking tokens: {max_thinking_tokens}")
    # print(f"Temperature: {temperature}")
    response = make_api_request(
        prompt=prompt,
        api_provider=api_provider,
        model_str=model_str,
        api_key=resolved_key,
        max_tokens=max_tokens,
        max_thinking_tokens=max_thinking_tokens,
        temperature=temperature,
        request_info={"pipeline_stage": "score_diversity"},
        logging_level="DEBUG" if verbose else "INFO",
    )

    # Parse the response for the count
    # Look for "COUNT: <int>" at the end
    match = re.search(r"COUNT:\s*(\d+)", response.strip().split('\n')[-1])
    if match:
        return int(match.group(1))
    
    # Fallback: try to find it anywhere
    match = re.search(r"COUNT:\s*(\d+)", response)
    if match:
        return int(match.group(1))

    # Fallback 2: Assume the last line is just the number if it's a digit
    last_line = response.strip().split('\n')[-1].strip()
    if last_line.isdigit():
        return int(last_line)

    print(f"Warning: Could not parse diversity score from response:\n{response}")
    return 0
    