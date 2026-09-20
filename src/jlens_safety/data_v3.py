"""Larger fresh-run data protocol. Selection never consults model outcomes."""

from __future__ import annotations

from collections import Counter
import ast
import csv
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from .common import atomic_json, digest, file_hash
from .data import fetch_csv, normalized, stratified_subset, validate_rows

TYPES = (
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
)


def wild_rows(records, evaluation=False, invalid_training_rows=None):
    result = []
    for i, r in enumerate(records):
        kind = r.get("data_type", "")
        if kind not in TYPES:
            raise ValueError(f"Unknown WildJailbreak data_type: {kind!r}")
        prompt = r.get(
            "adversarial" if kind.startswith("adversarial") else "vanilla", ""
        ).strip()
        completion = r.get("completion", "").strip()
        if (
            not evaluation
            and invalid_training_rows is not None
            and (not prompt or not completion)
        ):
            invalid_training_rows.append(
                dict(
                    id=f"wj-train-{i}",
                    category=kind,
                    reason=(
                        "missing_prompt"
                        if not prompt
                        else "missing_reference_completion"
                    ),
                )
            )
            continue
        if not prompt:
            raise ValueError("Missing WildJailbreak prompt")
        base = r.get("vanilla", "").strip() or prompt
        if not evaluation and not completion:
            raise ValueError("Coefficient fitting requires reference completions")
        result.append(
            dict(
                id=f'wj-{"eval" if evaluation else "train"}-{i}',
                prompt=prompt,
                base_prompt=base,
                group="request-" + digest(normalized(base))[:24],
                harmful=int(kind.endswith("_harmful")),
                category=kind,
                corpus="wildjailbreak",
                answer=None,
                completion=completion,
                split="test" if evaluation else None,
                group_source=(
                    "vanilla" if r.get("vanilla", "").strip() else "prompt_only"
                ),
            )
        )
    return result


def link_request_groups(rows):
    """Union exact normalized prompt/base matches across corpora and variants.

    Does not claim semantic deduplication; the manifest discloses that limit.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    seen = {}
    for r in rows:
        group = find(r["group"])
        for text in (r["prompt"], r.get("base_prompt", r["prompt"])):
            key = normalized(text)
            if key in seen:
                other = find(seen[key])
                parent[find(group)] = other
            seen[key] = group
    return [{**r, "group": find(r["group"])} for r in rows]


def select_training(rows, config):
    c, seed = config["data"], config["experiment"]["seed"]
    heldout = {r["group"] for r in rows if r["split"] == "test"}
    buckets = {(s, t): [] for s in ("train", "validation") for t in TYPES}
    seen, excluded = set(), Counter()
    for r in rows:
        if r["split"] == "test":
            continue
        if r["group"] in heldout:
            excluded["overlap_with_test_group"] += 1
            continue
        key = normalized(r["prompt"])
        if key in seen:
            excluded["duplicate_training_prompt"] += 1
            continue
        seen.add(key)
        value = int(digest([seed, r["group"]])[:16], 16) / 16**16
        split = "validation" if value < c["validation_group_fraction"] else "train"
        buckets[split, r["category"]].append({**r, "split": split})
    selected = [r for r in rows if r["split"] == "test"]
    for (split, kind), values in sorted(buckets.items()):
        random.Random(digest([seed, split, kind])).shuffle(values)
        count = c["train_per_type" if split == "train" else "validation_per_type"]
        if len(values) < count:
            raise ValueError(
                f"Only {len(values)} eligible {split}/{kind}; require {count}"
            )
        selected.extend(values[:count])
    return selected, dict(excluded)


def capped_test_groups(rows, config):
    """Deterministic, outcome-blind test reduction; never split request groups."""
    caps = config["data"].get("test_caps")
    if not caps:
        return rows
    groups = {}
    for row in rows:
        if row["split"] == "test":
            groups.setdefault(row["group"], []).append(row)
    totals = Counter()
    chosen = set()
    for group in sorted(
        groups, key=lambda key: digest([config["experiment"]["seed"], "test-cap", key])
    ):
        counts = Counter(f'{r["corpus"]}/{int(r["harmful"])}' for r in groups[group])
        if any(totals[key] + value > caps.get(key, 0) for key, value in counts.items()):
            continue
        chosen.add(group)
        totals.update(counts)
    if any(totals[key] < max(1, int(cap * 0.8)) for key, cap in caps.items()):
        raise ValueError(f"Test caps could not be met by whole groups: {dict(totals)}")
    return [r for r in rows if r["split"] != "test" or r["group"] in chosen]


def add_harmbench_attacks(behaviors, config):
    path = config["data"].get("harmbench_attack_file")
    mode = config["data"].get("harmbench_attack_mode", "file")
    attack_hash = None
    if mode == "published_human_jailbreaks":
        url = f'https://raw.githubusercontent.com/centerforaisafety/HarmBench/{config["data"]["harmbench_revision"]}/baselines/human_jailbreaks/jailbreaks.py'
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        templates = parse_published_templates(raw.decode())
        # Published Python literals include escaped surrogate pairs. Decode
        # those to Unicode scalars; fail on unpaired surrogates.
        templates = [
            text.encode("utf-16", "surrogatepass").decode("utf-16")
            for text in templates
        ]
        count = config["data"]["harmbench_attacks_per_behavior"]
        if not 1 <= count <= len(templates):
            raise ValueError("Invalid published attack count")
        rng = random.Random(config["data"]["harmbench_attack_seed"])
        cases = {}
        for r in behaviors:
            rng.shuffle(templates)
            cases[r["behavior_id"]] = [
                text + "\n\n" + r["behavior"] for text in templates[:count]
            ]
        attack_hash = digest(
            dict(source_sha256=hashlib.sha256(raw).hexdigest(), cases=cases)
        )
    elif mode != "file":
        raise ValueError("Unknown harmbench_attack_mode")
    elif not path:
        if config["data"].get("require_harmbench_attacks", True):
            raise ValueError(
                "Supply harmbench_attack_file and harmbench_attack_source before prepare; "
                "never silently replace adversarial evaluation with direct requests"
            )
        return [], None
    if not config["data"].get("harmbench_attack_source"):
        raise ValueError("Record the published test-case source attribution")
    if mode == "file":
        cases = json.loads(Path(path).read_text())
        attack_hash = file_hash(path)
    result = []
    for r in behaviors:
        values = cases.get(r["behavior_id"])
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(v, str) and v.strip() for v in values)
        ):
            raise ValueError(
                f'Missing/invalid adversarial test cases for {r["behavior_id"]}'
            )
        for i, text in enumerate(values):
            result.append(
                {
                    **r,
                    "id": r["id"] + f"-attack-{i}",
                    "prompt": text,
                    "corpus": "harmbench_adversarial",
                }
            )
    return result, attack_hash


def parse_published_templates(source):
    """Read a literal benchmark list without executing downloaded Python."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "JAILBREAKS" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(x, str) and x.strip() for x in value)
            ):
                raise ValueError("Invalid published template list")
            return value
    raise ValueError("Published template list not found")


def prepare(config, run):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    from transformers import AutoTokenizer

    d, hashes = config["data"], {}
    # Fail early on missing test cases, before the potentially large gated download.
    if (
        d.get("harmbench_attack_mode", "file") == "file"
        and d.get("require_harmbench_attacks", True)
        and not d.get("harmbench_attack_file")
    ):
        raise ValueError(
            "Set data.harmbench_attack_file and source; see docs/JLENS_SIGNED_TOKENS.md"
        )
    wj, invalid_training_rows = {}, []
    for split in ("train", "eval"):
        path = hf_hub_download(
            d["wildjailbreak_repo"],
            f"{split}/{split}.tsv",
            repo_type="dataset",
            revision=d["wildjailbreak_revision"],
        )
        hashes["wildjailbreak_" + split] = file_hash(path)
        with open(path, newline="", encoding="utf-8") as handle:
            wj[split] = wild_rows(
                csv.DictReader(handle, delimiter="\t"),
                split == "eval",
                invalid_training_rows if split == "train" else None,
            )
    atomic_json(run / "invalid_training_rows.json", invalid_training_rows)
    xs, hashes["xstest"] = fetch_csv(
        f'https://raw.githubusercontent.com/paul-rottger/xstest/{d["xstest_revision"]}/xstest_prompts.csv'
    )
    hb, hashes["harmbench"] = fetch_csv(
        f'https://raw.githubusercontent.com/centerforaisafety/HarmBench/{d["harmbench_revision"]}/data/behavior_datasets/harmbench_behaviors_text_test.csv'
    )
    test = list(wj["eval"])
    for r in xs:
        if r["label"] not in ("safe", "unsafe"):
            raise ValueError("Unexpected XSTest label")
        test.append(
            dict(
                id="xs-" + r["id"],
                prompt=r["prompt"],
                base_prompt=r["prompt"],
                group="xs-focus-" + normalized(r["focus"]),
                harmful=int(r["label"] == "unsafe"),
                split="test",
                category=r["type"],
                corpus="xstest",
                answer=None,
            )
        )
    behaviors = [
        dict(
            id="hb-" + r["BehaviorID"],
            behavior_id=r["BehaviorID"],
            behavior=r["Behavior"],
            prompt=r["Behavior"],
            base_prompt=r["Behavior"],
            group="hb-" + r["BehaviorID"],
            harmful=1,
            split="test",
            category=r["SemanticCategory"],
            corpus="harmbench_direct",
            answer=None,
        )
        for r in hb
        if r["FunctionalCategory"] == "standard"
    ]
    attacks, attack_hash = add_harmbench_attacks(behaviors, config)
    test.extend(behaviors + attacks)
    hashes["harmbench_attacks"] = attack_hash
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], revision=config["model"]["revision"]
    )

    def length(row):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    # No truncation or test exclusions. A too-short cap requires a new protocol.
    for r in test + wj["train"]:
        r["prompt_token_count"] = length(r)
    too_long = [
        r["id"]
        for r in test
        if r["prompt_token_count"] > config["model"]["max_prompt_tokens"]
    ]
    if too_long:
        raise ValueError(
            f"Test prompts exceed context cap; increase it in a NEW config: {too_long[:10]}"
        )
    train = [
        r
        for r in wj["train"]
        if r["prompt_token_count"] <= d["max_training_prompt_tokens"]
    ]
    linked = capped_test_groups(link_request_groups(train + test), config)
    rows, excluded = select_training(linked, config)
    excluded["training_pool_over_context"] = len(wj["train"]) - len(train)
    excluded["malformed_training_rows"] = len(invalid_training_rows)
    for split, original in [("validation", "train"), ("test", "test")]:
        path = hf_hub_download(
            "openai/gsm8k",
            f"main/{original}-00000-of-00001.parquet",
            repo_type="dataset",
            revision=d["gsm8k_revision"],
        )
        hashes["gsm8k_" + original] = file_hash(path)
        records = pd.read_parquet(path).to_dict("records")
        indices = list(range(len(records)))
        random.Random(config["experiment"]["seed"]).shuffle(indices)
        count = (
            d["gsm8k_validation"]
            if split == "validation"
            else d.get("gsm8k_test", len(records))
        )
        for i in indices[:count]:
            r = records[i]
            rows.append(
                dict(
                    id=f"gsm-{original}-{i}",
                    group=f"gsm-{original}-{i}",
                    prompt=r["question"]
                    + "\nGive your final numeric answer after ####.",
                    harmful=0,
                    split=split,
                    category="math",
                    corpus="gsm8k",
                    answer=r["answer"].split("####")[-1].strip().replace(",", ""),
                )
            )
    validate_rows(rows)
    atomic_json(
        run / "data_manifest.json",
        dict(
            rows_hash=digest(rows),
            sha256=hashes,
            counts=dict(
                Counter(f'{r["split"]}/{r["corpus"]}/{r["category"]}' for r in rows)
            ),
            exclusions=excluded,
            invalid_training_rows=invalid_training_rows,
            attack_source=d.get("harmbench_attack_source"),
            attack_unicode_normalization="Escaped UTF-16 surrogate pairs decoded; unpaired surrogates rejected",
            deduplication="Normalized exact prompt/base union plus supplied focus groups; semantic overlap may remain",
            prior_exposure="XSTest/HarmBench/GSM8K were previously examined; new generations are not new benchmark holdouts",
            training_pool_source="WildJailbreak train only; official eval is never used to fit or select",
        ),
    )
    tmp = run / "prompts.tmp"
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    tmp.replace(run / "prompts.jsonl")
    return rows


def subset(rows, split, per_type, gsm_count, seed):
    chosen = []
    for i, kind in enumerate(TYPES):
        pool = [
            r
            for r in rows
            if r["split"] == split
            and r["corpus"] == "wildjailbreak"
            and r["category"] == kind
        ]
        chosen.extend(stratified_subset(pool, per_type, seed + i))
    math = [r for r in rows if r["split"] == split and r["corpus"] == "gsm8k"]
    chosen.extend(stratified_subset(math, gsm_count, seed))
    return chosen
