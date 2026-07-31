import re
import os
import json
import argparse
import pandas as pd
from enum import Enum
from tqdm import tqdm
from google import genai
from pydantic import BaseModel
from utils import (
    parse_instruction, 
    check_instruction, 
    count_tag_types, 
    generate_conv_metadata, 
    reorg_data
)

tqdm.pandas()


class ClaimCitationPair(BaseModel):
    claim: str
    citation: list[int]


class AnswerType(str, Enum):
    support = "support"
    contradict = "contradict"
    irrelevant = "irrelevant"


class GroundingResult(BaseModel):
    reasoning: str
    answer: AnswerType


def llm_citation_claim_parsing(text: str, web_trace: dict, model_name: str = "gemini-2.5-flash") -> list:
    """Parses text chunks to extract claim statements associated with their respective inline citations."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"{parse_instruction}\n\n{text}"
    messages = [{"role": "user", "parts": [{"text": prompt}]}]

    # Backoff retry logic
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=messages,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[ClaimCitationPair],
                    'temperature': attempt * 0.05,  # Slightly raise temperature on failures
                    'max_output_tokens': 64000,
                }
            )
            
            if not response or not response.parsed:
                continue

            chunks = []
            for item in response.parsed:
                if isinstance(item, ClaimCitationPair):
                    citation_urls = []
                    for citation in item.citation:
                        key = f"[{citation}]"
                        if key in web_trace:
                            citation_urls.append(web_trace[key])
                    if citation_urls:
                        chunks.append((item.claim, citation_urls))
                else:
                    print(f"Unexpected structural item type returned: {type(item)}")
            
            return chunks

        except Exception as e:
            print(f"Error encountered parsing citations on attempt {attempt + 1}: {e}")
            continue

    return []


def resolve_citations(messages: list, web_traces: list) -> list:
    """
    Resolves source citations out of a historical list of multi-turn chat messages.
    Returns list records comprising:
      - 'prompt': The initial user query text
      - 'answer_chunks': list structure of gathered (claim_text, list_of_urls) tuples
    """
    results = []
    n_rounds = len(messages) // 2

    for round_idx in range(n_rounds):
        question_msg = messages[round_idx * 2]
        answer_msg = messages[round_idx * 2 + 1]
        
        question_text = question_msg.get("content", "")
        answer_text = answer_msg.get("content", "")

        # Skip evaluation if trace map is non-existent or no brackets exist in textual output
        if len(web_traces[round_idx]) == 0 or not re.search(r"\[(\d+)\]", answer_text):
            continue
            
        web_trace_dict = {}
        for marker, url in web_traces[round_idx]:
            web_trace_dict[str(marker)] = url
            
        chunks = llm_citation_claim_parsing(answer_text, web_trace_dict)
        results.append({
            "prompt": question_text,
            "answer_chunks": chunks
        })

    return results


def llm_misattribution_check(claims: list, web_content: str, source: str) -> dict:
    """Validates if individual target claims are structurally supported, contradicted, or irrelevant to sources."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = check_instruction.format(claims=claims, web_content=web_content, source=source)
    messages = [{"role": "user", "parts": [{"text": prompt}]}]

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[GroundingResult],
                    'temperature': attempt * 0.05,
                    'max_output_tokens': 64000,
                }
            )
            
            if not response or not response.parsed:
                continue
                
            if len(response.parsed) != len(claims):
                print(f"Warning: validation array length mismatch ({len(response.parsed)} vs {len(claims)})")
                continue
                
            return {
                "claims": claims,
                "attribute_label": [item.answer.value for item in response.parsed],
                "attribute_reasoning": [item.reasoning for item in response.parsed]
            }
        except Exception as e:
            print(f"Grounding verification exception on attempt {attempt + 1}: {e}")
            continue

    return {
        "claims": claims,
        "attribute_label": ["Failed"] * len(claims),
        "attribute_reasoning": ["Failed"] * len(claims)
    }


def append_record(output_path: str, record: dict) -> None:
    """Appends validation progress metadata records line-by-line using a clean context manager."""
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate verification claims and ground inline citations using Gemini.")
    parser.add_argument("--url_fname", type=str, default="data/scraped_urls.jsonl", help="Path to scraped URLs database.")
    parser.add_argument("--dataset_fname", type=str, default="data/subset.jsonl", help="Path to evaluation source subset dataset.")
    parser.add_argument("--intermediate_fname", type=str, default="data/resolved_citations.jsonl", help="Target output tracking file for path resolution.")
    parser.add_argument("--llm_judge_fname", type=str, default="data/judged_citations.jsonl", help="Staging output log tracking judge returns.")
    parser.add_argument("--output_fname", type=str, default="data/citation_attribution_data.jsonl", help="Final downstream telemetry path.")
    parser.add_argument("--cmd", type=str, required=True, choices=["citation_resolve", "citation_check"], help="Pipeline execution command segment context.")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("System environment flag validation failure: GEMINI_API_KEY is not configured.")

    # RUN STATE 1: Citation Resolution Pipeline
    if args.cmd == "citation_resolve":
        print("Starting resolution steps: reading input source records...")
        dataset = pd.read_json(args.dataset_fname, lines=True)

        # Drop fallback data intent classifications
        if "primary_intent" in dataset.columns:
            dataset = dataset[dataset["primary_intent"] != "Other"]
            
        dataset["web_trace_a"] = dataset["system_a_metadata"].apply(lambda x: x.get("web_search_trace", []) if isinstance(x, dict) else [])
        dataset["web_trace_b"] = dataset["system_b_metadata"].apply(lambda x: x.get("web_search_trace", []) if isinstance(x, dict) else [])
        
        print("Running distributed processing over text content matrices...")
        dataset["resolved_a"] = dataset.progress_apply(
            lambda row: resolve_citations(row["messages_a"], row["web_trace_a"]), axis=1
        )
        dataset["resolved_b"] = dataset.progress_apply(
            lambda row: resolve_citations(row["messages_b"], row["web_trace_b"]), axis=1
        )
        
        dataset.to_json(args.intermediate_fname, lines=True, orient="records")
        print(f"Resolution tasks complete. Staged items tracked into: {args.intermediate_fname}")

    # RUN STATE 2: Attribution Checking Pipeline
    elif args.cmd == "citation_check":
        url_df = pd.read_json(args.url_fname, lines=True)
        url_dict = dict(zip(url_df["url"], url_df["text"]))
        
        dataset = pd.read_json(args.intermediate_fname, lines=True)
        citation_entries = reorg_data(dataset)
        
        print(f"Beginning validation loop iterations over {len(citation_entries)} global reference footprints...")
        for url, entry in tqdm(citation_entries.items(), desc="Checking Attribution"):
            if url not in url_dict:
                print(f"Skipping lookup trace mismatch: {url} couldn't be located inside scraped URL indexes.")
                continue
                
            result = llm_misattribution_check(entry["claims"], url_dict[url], entry["source"])
            entry["result"] = result
            append_record(args.llm_judge_fname, entry)

        # Final structural assembly transformations
        print("Compiling global telemetry scores, generating execution summary reports...")
        input_df = pd.read_json(args.llm_judge_fname, lines=True)
        
        # Safe structural dictionary expanding mapping
        tag_counts_df_a = pd.DataFrame(input_df['claim_attribution_tags_a'].apply(count_tag_types).tolist())
        tag_counts_df_b = pd.DataFrame(input_df['claim_attribution_tags_b'].apply(count_tag_types).tolist())
        
        tag_counts_df_a.columns = ['support_count_a', 'contradict_count_a', 'irrelevant_count_a', 'total_count_a']
        tag_counts_df_b.columns = ['support_count_b', 'contradict_count_b', 'irrelevant_count_b', 'total_count_b']

        final_df = pd.concat([input_df, tag_counts_df_a, tag_counts_df_b], axis=1)
        final_df["conv_metadata"] = final_df.apply(generate_conv_metadata, axis=1)
        
        final_df.to_json(args.output_fname, lines=True, orient="records")
        print(f"Execution run complete. Pipeline metadata written cleanly to target endpoint location: {args.output_fname}")
