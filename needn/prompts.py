"""LLM-as-predictor prompts (text-only f0 and graph-augmented f1).

Layout: [system] [task description + answer options] [node text] [neighbour context] [question].
The long, node-independent part (incl. the option list, up to 40 classes on arxiv) comes first so that vLLM's
prefix cache can share it across all prompts of a dataset; the question is repeated at the end.
"""
import string

LETTERS = list(string.ascii_uppercase) + list(string.ascii_lowercase)
SYSTEM = "You are an expert at categorising items of a network. Answer with a single option letter."


def trunc(t, n):
    w = (t or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def options(names):
    # A-Z then a-z (single tokens for the Qwen / Llama tokenizers); only arXiv (40 classes) needs lowercase letters
    assert len(names) <= len(LETTERS), f"{len(names)} classes > {len(LETTERS)} single-character option letters"
    return "\n".join(f"{LETTERS[i]}. {n}" for i, n in enumerate(names))


def nodes_(m, n=2):
    """node noun, plural from the config ('nodes') -- appending 's' breaks e.g. 'Reddit user (their recent posts)'"""
    return m["node"] if n == 1 else m.get("nodes") or m["node"] + "s"


def a_(phrase):
    vowel = phrase[:1].lower() in "aeiou" and not phrase.lower().startswith(("uni", "use", "eu", "one"))
    return ("an " if vowel else "a ") + phrase


def links_(m):
    """edge phrase, e.g. 'citation links'; config 'links' overrides it (avoids 'hyperlink links')"""
    return m.get("links") or f"{m['edge']} links"


def cap(s):
    return s[:1].upper() + s[1:]  # str.capitalize() would lower-case the rest ('arXiv' -> 'Arxiv')


def _head(g, what):
    m = g["meta"]
    return (f"You will be shown {what} from {m['domain']} and must determine its {m['task']}.\n"
            f"Options:\n{options(g['label_names'])}\n\n")


def f0(g, v, cfg, text=None):
    """text-only prompt; text overrides the node text (used with 'N/A' for contextual calibration)"""
    m, pc = g["meta"], cfg["prompt"]
    body = text if text is not None else trunc(g['texts'][v], pc['target_words'])
    u = (_head(g, a_(m['node'])) +
         f"{cap(m['node'])}: {body}\n\n"
         f"Question: what is its {m['task']}? Answer with the option letter only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]


def self_need(g, v, cfg):
    """LLM self-assessment router (baseline): does the model think it needs the neighbours?"""
    m, pc = g["meta"], cfg["prompt"]
    u = (f"The following is {a_(m['node'])} from {m['domain']}. Possible values of its {m['task']}: "
         f"{', '.join(g['label_names'])}.\n\n"
         f"{cap(m['node'])}: {trunc(g['texts'][v], pc['target_words'])}\n\n"
         f"Can its {m['task']} be determined confidently from this text alone, without looking at the {nodes_(m)} "
         "it is connected to?\nA. Yes, the text alone is sufficient\nB. No, information about connected items would help\n\n"
         "Answer with the option letter only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]


def f1(g, v, nbrs, cfg, hints=None, summary=None, nbr_texts=None):
    m, pc = g["meta"], cfg["prompt"]
    if summary is not None:
        ctx = f"Summary of the {nodes_(m)} the target is connected to by {links_(m)}: {summary}"
    else:
        lines = []
        for i, u in enumerate(nbrs):
            h = f" (likely {m['task']}: {g['label_names'][hints[i]]})" if hints is not None else ""
            body = nbr_texts[i] if nbr_texts is not None else trunc(g['texts'][u], pc['neighbor_words'])
            lines.append(f"[{i + 1}]{h} {body}")
        ctx = (f"The target is connected by {links_(m)} to the following {len(nbrs)} {nodes_(m, len(nbrs))}:\n"
               + "\n".join(lines) if lines else f"The target has no {links_(m)}.")
    # the target is called "the target <node>" -- an earlier wording ("the first <node>") biased models towards
    # option A
    u = (_head(g, f"{a_(m['node'])} together with {nodes_(m)} it is linked to") +
         f"Target {m['node']}: {trunc(g['texts'][v], pc['target_words'])}\n\n{ctx}\n\n"
         f"Question: what is the {m['task']} of the target {m['node']}? Use the connected {nodes_(m)} only as "
         f"additional context. Answer with the option letter only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]


def summarize(g, nbrs, cfg):
    m, pc = g["meta"], cfg["prompt"]
    lines = "\n".join(f"[{i + 1}] {trunc(g['texts'][u], pc['neighbor_words'])}" for i, u in enumerate(nbrs))
    return [{"role": "user", "content": f"Summarize in 2-3 sentences what the following {nodes_(m)} are about "
                                        f"(focus on their {m['task']}):\n{lines}"}]


def graphicl(g, v, nbrs, demos, cfg):
    """GraphICL-style training-free baseline (Sun et al., 2025): structure-aware in-context learning with labelled
    demonstrations (the most similar training nodes, with their labels) and the target's neighbours (their label is
    shown when it is a training label)."""
    m, pc = g["meta"], cfg["prompt"]
    names, train = g["label_names"], g["train"]
    ex = "\n\n".join(f"Example {i + 1}: {trunc(g['texts'][u], pc['neighbor_words'])}\n"
                      f"Answer: {LETTERS[int(g['y'][u])]} ({names[int(g['y'][u])]})" for i, u in enumerate(demos))
    lines = []
    for i, u in enumerate(nbrs):
        h = f" (known {m['task']}: {names[int(g['y'][u])]})" if train[u] else ""
        lines.append(f"[{i + 1}]{h} {trunc(g['texts'][u], pc['neighbor_words'])}")
    ctx = (f"The target is connected by {links_(m)} to the following {len(nbrs)} {nodes_(m, len(nbrs))}:\n"
           + "\n".join(lines) if lines else f"The target has no {links_(m)}.")
    u = (_head(g, f"{a_(m['node'])} together with {nodes_(m)} it is linked to") +
         f"Labelled examples from the same graph:\n{ex}\n\n"
         f"Target {m['node']}: {trunc(g['texts'][v], pc['target_words'])}\n\n{ctx}\n\n"
         f"Question: what is the {m['task']} of the target {m['node']}? Use the examples and the connected "
         f"{nodes_(m)} only as additional context. Answer with the option letter only.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]
