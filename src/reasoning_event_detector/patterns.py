"""
VENDORED, FROZEN COPY. Do not edit here.

Source of truth:
  hawkes_pilot/artifacts/event_detection_2026_05_12/detectors.py
  sha256 f7a1a89d81397758e580028ae24de66970365ea195f1b5ac93476ac53bf6246c
  copied verbatim on 2026-08-11 (only this provenance block was prepended).

The regexes are byte-identical to the canonical layer so that the events this
package emits are the same objects the project's measurements were made on.
Any change here silently invalidates every number in MODEL_CARD.md.

Public surface used by this package: DOMAIN_M_PATTERNS, DOMAIN_R_PATTERNS,
DOMAIN_BY_BENCHMARK, detect_events(text, domain), char_to_token_pos(text, pos).

────────────────────────────────────────────────────────────────────────────
Event detectors v2 (rewritten 2026-05-12 after H0 validation).

Changes from v1:
  Domain M:
    - verify: rewritten to accept "let's check", "checking <noun>",
      "let me re-check", "let me just quickly check", "let me verify",
      and several other variants v1 missed.
    - branch: tightened to require COLON after "option N" so that decimals
      like "option 16.9%" no longer fire as branch FP.
    - subgoal_done: added "Conclusion:", "Conclusion for X:", "Result:",
      "in conclusion", "we conclude that", "it follows that".
    - commit: added "Final decision:", "Final answer should be",
      "in conclusion, the answer/result".
    - backtrack, failed_attempt: minimal changes (recall already ~1.0 in v1).

  Domain R:
    - evidence_merge: loosened adjacency so up to 3 inserted words
      ("previous", "first", "for X") no longer break the match. Added
      verb conjugations (-s, -ed, -ing), more verbs ("provide", "return",
      "hint", "imply"), more sources ("snippet", "reference"), and the
      "it says/states/...:" inline form.
    - reasoning_step -> RENAMED to `decomposition`. The v1 regex captured
      planning markers ("Step 1:", numbered lists, "First,") which are
      decomposition events, not in-line reasoning steps. Renaming is
      honest about what we actually detect.
    - query_rewrite: rewritten to catch the patterns v1 missed completely
      ("I'll search specifically", "do one more search", "let me try to
      find", "let's try to search", "search again", "another search", etc.)
      v1 recall on this type was 0.00.
    - contradiction: tightened to require source/page mention nearby
      ("however, the page" rather than bare "however,"). This kills the
      massive FP storm on plain narrative "however," (in gemma musique,
      1744 of 1798 contradiction events were just "however,").
    - retrieval_call: unchanged (F1 = 1.00 on validation sample).

  Cross-domain:
    - DOMAIN_BY_BENCHMARK and detect_events()/char_to_token_pos()
      unchanged (same API).

K stays at 6 in both domains, so downstream likelihood models do not
require dimensional changes; only the renamed type 'decomposition'
needs to be reflected in likelihoods.py DOMAIN_TYPES.
"""
import re


# ---------------------------------------------------------------------------
# Domain M: math / science / puzzles  (math500, gpqa_diamond, hle)
# ---------------------------------------------------------------------------

DOMAIN_M_PATTERNS = {

    'backtrack': re.compile(
        r"(?:"
        r"\bwait[,.](?:\s|$)|"
        r"\bactually[,.](?:\s|$)|"
        r"\bhmm[,.]|"
        r"\bhold on[,.]|"
        r"\blet me reconsider|"
        r"\bon second thought|"
        r"\boh wait|"
        r"\bbut wait|"
        r"\bwait,?\s+no|"
        r"\bwait,?\s+actually|"
        r"\bi was wrong|"
        r"\bi made (?:a |an )?mistake"
        r")",
        re.IGNORECASE,
    ),

    'verify': re.compile(
        r"(?:"
        # IMPERATIVE: "let me / let's / I'll / I should + [adv]? + [re-]? + [double-]? + verb"
        # verbs: check / verify / confirm / recompute / recalculate
        r"\b(?:let me|let'?s|i'?ll|i should|i need to|i want to|we should|we need to)"
        r"(?:\s+(?:also|just|quickly|briefly|first|now|carefully|definitely|again|once more|really|then))?"
        r"\s+(?:re[- ]?)?(?:double[- ]?)?"
        r"(?:check|verify|confirm|recompute|re[- ]?calculate)\b"
        r"|"
        # STANDALONE: double-check / verify with conjugation
        r"\b(?:re[- ]?)?double[- ]?check(?:ing|ed|s)?\b"
        r"|"
        r"\b(?:re[- ]?)?verify(?:ing|ied|ies)?\b"
        r"|"
        # "Check:" or "Check." at sentence-like position
        r"\b(?:re[- ]?)?check(?:ing)?[:.]"
        r"|"
        # "Checking <noun-phrase>" (mid-sentence verification with explicit object)
        r"\bchecking\s+(?:if|that|whether|the|all|each|every|any|option|some|both|further|more)\b"
        r"|"
        # "Sanity check"
        r"\bsanity[- ]check\b"
        r"|"
        # AUXILIARY: "to verify / to confirm / to make sure / to be sure / to double-check"
        r"\bto (?:verify|confirm|make sure|be sure|double[- ]?check)\b"
        r")",
        re.IGNORECASE,
    ),

    'branch': re.compile(
        r"(?:"
        r"\bcase\s+\d+[:.]|"
        r"\balternatively[,.]|"
        # NEW: require a colon strictly (avoids matching decimal "option 16.9%")
        r"\boption\s+\d+:|"
        r"\bpossibility\s+\d+[:.]|"
        r"\bscenario\s+\d+[:.]|"
        r"\bon the other hand[,.]|"
        r"\banother approach\b|"
        r"\banother way\b|"
        r"\beither\s+\w+\s+or\b"
        r")",
        re.IGNORECASE,
    ),

    'subgoal_done': re.compile(
        r"(?:"
        r"\btherefore[,.]|"
        r"\bthus[,.]|"
        r"\bhence[,.]|"
        r"\bso we (?:have|get|conclude|obtain)\b|"
        r"\bthis (?:gives|yields|means|implies)\b|"
        r"\bso the\s+\w+\s+is\b|"
        # NEW: explicit conclusion markers
        r"\bconclusion(?:\s+for\s+\w+)?\s*:|"
        r"\bresult\s*:|"
        r"\bfinal result\s*:|"
        r"\bin conclusion[,.]|"
        r"\bto conclude[,.]|"
        r"\bwe conclude that\b|"
        r"\bit follows that\b"
        r")",
        re.IGNORECASE,
    ),

    'failed_attempt': re.compile(
        r"(?:"
        r"\bthis (?:doesn'?t|does not) work\b|"
        r"\bthat (?:doesn'?t|does not) work\b|"
        r"\bdead end\b|"
        r"\b(?:doesn'?t|does not) make sense\b|"
        r"\bthat'?s (?:wrong|incorrect|not right)\b|"
        r"\bscratch that\b|"
        r"\bcan'?t be (?:right|correct|the case)\b|"
        r"\bthat won'?t work\b|"
        r"\bno,? that'?s not\b|"
        r"\bthat'?s impossible\b|"
        r"\bcontradiction[,.]"
        r")",
        re.IGNORECASE,
    ),

    'commit': re.compile(
        r"(?:"
        r"\bthe answer is\b|"
        r"\bso the answer is\b|"
        r"\bthe final answer is\b|"
        r"\bfinal answer\s*[:.]|"
        # NEW
        r"\bfinal decision\s*[:.]|"
        r"\bfinal answer should be\b|"
        r"\bin conclusion,?\s+(?:the answer|the result)|"
        r"\\boxed\{|"
        r'"concise_answer"|'
        r"\banswer:\s"
        r")",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Domain R: retrieval  (hotpotqa, musique)
# ---------------------------------------------------------------------------

DOMAIN_R_PATTERNS = {

    'retrieval_call': re.compile(
        r"(?:"
        r"<wikipedia_search[:_]|"
        r"<wikipedia_page[:_]|"
        r"<web_search[:_]|"
        r"<search[:_]|"
        r"<tool_call|"
        r"<retrieve[:_]|"
        r"<fetch[:_]"
        r")",
        re.IGNORECASE,
    ),

    'evidence_merge': re.compile(
        r"(?:"
        # "the [0-3 words] search result(s) VERB"
        # e.g. "the first search result for 'X' indicates"
        r"\bthe (?:\w+\s+){0,3}search results? "
        r"(?:indicate|show|reveal|suggest|say|tell|confirm|mention|state|provide|return|hint|point|imply|give)"
        r"(?:s|ed|ing)?\b"
        r"|"
        # "from (the/this/our/...) [0-2 words] search(es) (results?)?"
        r"\bfrom (?:the |this |these |my |our |previous |earlier |all (?:the |my )?)?"
        r"(?:\w+\s+){0,2}search(?:es)?(?:\s+results?)?[:,.\s]"
        r"|"
        # "based on [0-3 words] (search|information|...|wikipedia|snippet)"
        r"\bbased on (?:the |this |our |my )?(?:\w+\s+){0,3}"
        r"(?:search|information|result|page|article|evidence|wikipedia|snippet|source|find|hint|fact)\b"
        r"|"
        # "according to (the/this/...) [0-2 words] (page|article|...)"
        r"\baccording to (?:the |this |that |another |my |our )?(?:\w+\s+){0,2}"
        r"(?:page|article|search|source|wikipedia|snippet|result)\b"
        r"|"
        # "the/another/this/that [0-3 words] (page|article|...|reference) VERB"
        r"\b(?:the|another|this|that) (?:\w+\s+){0,3}"
        r"(?:page|article|source|wikipedia|snippet|result|reference)\s+"
        r"(?:says|states|mentions|notes|indicates|confirms|reveals|suggests|shows|tells|hints)"
        r"(?:s|ed|ing)?\b"
        r"|"
        # "it (says|states|...) :' \"" - introducing a quote
        r"\bit (?:says|states|mentions|notes|indicates|confirms|reveals|tells)\s*[:'\"]"
        r"|"
        # "the wikipedia (page|article|summary|entry)"
        r"\bthe wikipedia (?:page|article|summary|entry)\b"
        r"|"
        # "the/another snippet/result for/from X (verb)"
        r"\b(?:the|another)\s+(?:snippet|result|find|reference)\s+(?:from|for|of|on)\s+"
        r"['\"]?\w+['\"]?\s+(?:says|states|mentions|notes|indicates|confirms|reveals|suggests)\b"
        r"|"
        # "search/snippet result(s) provide/give strong/weak/clear evidence"
        r"\b(?:search|snippet) results?\s+(?:provide|give|offer)\s+(?:strong|weak|clear|good|some)\s+(?:evidence|hint|indication|sign)"
        r")",
        re.IGNORECASE,
    ),

    'contradiction': re.compile(
        r"(?:"
        # NEW: require source mention right after "however"
        r"\bhowever,?\s+(?:the|this|that|another)\s+(?:page|source|article|result|snippet|search|wikipedia|reference)\b"
        r"|"
        # "but (the|this|that|another) (page|source|...)"
        r"\bbut (?:the|this|that|another) (?:page|source|article|result|snippet|search|wikipedia|reference)\b"
        r"|"
        # explicit contradiction/conflict words (still cheap and clean)
        r"\bcontradict(?:s|ion|ory|ed|ing)?\b"
        r"|"
        r"\bconflict(?:s|ed|ing)? with\b"
        r"|"
        # "in contrast, X [source]"
        r"\bin contrast,?\s+(?:the|this|that|another|one)\b"
        r"|"
        r"\bin opposition\b"
        r"|"
        r"\binconsistent\b"
        r"|"
        r"\bdisagree(?:s|d|ment)?\b"
        r"|"
        # "X (does|do) not (match|agree|fit)"
        r"\b(?:does|do) (?:not|n't) (?:match|agree|fit)\b"
        r")",
        re.IGNORECASE,
    ),

    # RENAMED from 'reasoning_step'. Captures planning/decomposition events
    # (numbered steps, "Step N:", "First,/Then,/Next,/Finally,", plan headers).
    'decomposition': re.compile(
        r"(?:"
        r"\bstep\s+\d+[:.]\s|"
        # numbered list item at line start
        r"(?:^|\n)\s*\d+[.)]\s+\w|"
        # sequence words near sentence start (after newline or after sentence boundary)
        r"(?:^|\n|[.!?]\s)\s*(?:first|second|third|fourth|fifth|next|then|finally)[,:]\s|"
        # plan / criteria / outline / approach / steps headers
        r"\bplan(?:\s+\w+)?\s*[:.]|"
        r"\bcriteria\s*[:.]|"
        r"\bsteps?\s*[:.]\s|"
        r"\bsubgoals?\s*[:.]|"
        r"\boutline\s*[:.]|"
        r"\bapproach\s*[:.]"
        r")",
        re.IGNORECASE | re.MULTILINE,
    ),

    'query_rewrite': re.compile(
        r"(?:"
        # imperative search/lookup/query rewriting
        r"\b(?:let me|let'?s|i'?ll|i (?:should|need to|want to))"
        r"(?:\s+(?:just|also|quickly|briefly|now))?"
        r"\s+(?:try (?:to )?(?:search|find|look|fetch|query|consult)|search|look|query|consult)\b"
        r"|"
        # "search [adverb]"
        r"\bsearch(?:ing)? (?:specifically|differently|again|more (?:specifically|broadly)|for the|for a)\b"
        r"|"
        # "do (one more|another|a different) search/query/lookup"
        r"\bdo (?:one more|another|a different) (?:search|query|lookup)\b"
        r"|"
        # "(another|different|new|one more) (search|query|lookup|attempt)"
        r"\b(?:another|different|new|one more) (?:search|query|lookup|attempt)\b"
        r"|"
        # refine / reformulate / rephrase (the|my|our) search/query
        r"\b(?:refine|reformulate|rephrase) (?:the |my |our )?(?:search|query)\b"
        r"|"
        # "try [a|another] [different] (spelling|query|wording|approach|search term)"
        r"\btry (?:a |another |a different )?(?:different )?(?:spelling|query|wording|approach|search term)\b"
        r"|"
        # "search again"
        r"\bsearch again\b"
        r"|"
        # "let me/let's look (up|for|elsewhere)"
        r"\b(?:let me|let'?s) look (?:up|for|elsewhere)\b"
        r"|"
        # "I'll (search|look|check) (for|up) ..."
        r"\bi'?ll (?:search|look|check) (?:for|up)\b"
        r")",
        re.IGNORECASE,
    ),

    'commit': re.compile(
        r"(?:"
        r"\bbased on (?:the )?(?:evidence|information|facts?|search|results?)|"
        r"\bthe answer is\b|"
        r"\bso the answer is\b|"
        r"\bthe final answer is\b|"
        r"\bfinal answer\s*[:.]|"
        r"\bfinal decision\s*[:.]|"
        r"\bfinal answer should be\b|"
        r"\bin conclusion,?\s+(?:the answer|the result)|"
        r"\btherefore,?\s+(?:the answer|the result)|"
        r'"concise_answer"'
        r")",
        re.IGNORECASE,
    ),

    # NEW (2026-05-12, K extended 6->10): metacognitive events shared with
    # Domain M, but reused here because retrieval traces have plenty of them.
    'backtrack': re.compile(
        r"(?:"
        r"\bwait[,.](?:\s|$)|"
        r"\bactually[,.](?:\s|$)|"
        r"\bhmm[,.]|"
        r"\bhold on[,.]|"
        r"\blet me reconsider|"
        r"\bon second thought|"
        r"\boh wait|"
        r"\bbut wait|"
        r"\bwait,?\s+no|"
        r"\bwait,?\s+actually|"
        r"\bi was wrong|"
        r"\bi made (?:a |an )?mistake"
        r")",
        re.IGNORECASE,
    ),

    'verify': re.compile(
        r"(?:"
        # IMPERATIVE: "let me / let's / I'll / I should + [adv]? + [re-]? + [double-]? + verb"
        r"\b(?:let me|let'?s|i'?ll|i should|i need to|i want to|we should|we need to)"
        r"(?:\s+(?:also|just|quickly|briefly|first|now|carefully|definitely|again|once more|really|then))?"
        r"\s+(?:re[- ]?)?(?:double[- ]?)?"
        r"(?:check|verify|confirm|recompute|re[- ]?calculate)\b"
        r"|"
        # STANDALONE
        r"\b(?:re[- ]?)?double[- ]?check(?:ing|ed|s)?\b"
        r"|"
        r"\b(?:re[- ]?)?verify(?:ing|ied|ies)?\b"
        r"|"
        r"\b(?:re[- ]?)?check(?:ing)?[:.]"
        r"|"
        r"\bchecking\s+(?:if|that|whether|the|all|each|every|any|option|some|both|further|more)\b"
        r"|"
        r"\bsanity[- ]check\b"
        r"|"
        r"\bto (?:verify|confirm|make sure|be sure|double[- ]?check)\b"
        r")",
        re.IGNORECASE,
    ),

    # NEW: merged 'hypothesis' (speculation + disambiguation) and
    # 'memory_recall' (fact from weights). Cognitively unified as
    # "model uses internal state instead of grounding in retrieved evidence".
    # Built from 5 tiers; mixed case-sensitive / case-insensitive via (?i:...)
    # inline flags. Compiled without global IGNORECASE so that proper-name
    # patterns (Tier C of memory_recall) work correctly.
    'internal_use': re.compile(
        r"(?:"
        # === HYPOTHESIS markers ===
        r"(?i:\b(?:maybe|perhaps|possibly|presumably|seemingly)[,.\s])"
        r"|(?i:\bit\s+(?:seems|appears)\s+(?:that|to|like))"
        r"|(?i:\bcould\s+(?:it|that|this|the|a |an )\s+be\b)"
        r"|(?i:\b(?:might|may)\s+(?:be|have)\b)"
        r"|(?i:\b(?:i|we)\s+suspect\b)"
        r"|(?i:\b(?:i|we)\s+(?:guess|wonder)\b)"
        # disambiguation forms
        r"|(?i:\bwhich\s+(?:one|of|specific|particular|former|character|player|film|movie|song|album|book|company|club|team|series|version|year|date|person|member|episode))"
        r"|(?i:\bcould\s+be\s+multiple\b)"
        r"|(?i:\b(?:may|might|could)\s+(?:refer|mean)\s+to\b)"
        r"|(?i:\bthere\s+(?:are|may\s+be|might\s+be|could\s+be)\s+multiple\b)"

        # === MEMORY_RECALL TIER A: explicit recall markers ===
        r"|(?i:\b(?:we|i)\s+know\s+(?:that|typical|it|its|this|those|these|of|about|already)\b)"
        r"|(?i:\b(?:we|i)\s+(?:recall|remember)\b)"
        r"|(?i:\bif\s+(?:i|we)\s+(?:recall|remember)(?:\s+correctly)?\b)"
        r"|(?i:\bas\s+far\s+as\s+(?:i|we)\s+(?:know|recall|remember)\b)"
        r"|(?i:\bfrom\s+memory\b)"
        r"|(?i:\bfrom\s+what\s+(?:i|we)\s+(?:know|recall|remember)\b)"
        r"|(?i:\b(?:i|we)\s+(?:believe|think)\s+(?:that|it|this|the|a |an |she |he |they ))"
        r"|(?i:\bbut\s+(?:we|i)\s+know\b)"
        r"|(?i:\b(?:iirc|afaik)\b)"

        # === MEMORY_RECALL TIER B: parenthesized biographic metadata ===
        r"|(?i:\(born\s+(?:in\s+|on\s+)?[^)]{0,30}\d{4}[^)]*\))"
        r"|\(\s*\d{4}\s*[-–—]\s*\d{4}\s*\)"
        r"|(?i:\(\s*(?:singer|director|actor|actress|musician|writer|composer|footballer|painter|playwright|screenwriter|novelist|poet|philosopher|politician|scientist|engineer|architect|comedian|drummer|guitarist|founder|inventor|professor)[^)]*\))"

        # === MEMORY_RECALL TIER C: case-sensitive proper-name + verb ===
        # 'X Y was born in/on'
        r"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,3}\s+(?:was|were|is|are)\s+born\s+(?:in|on)\b"
        # 'X Y was a/an/the [role/group/work]'
        r"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,3}\s+(?:was|were|is|are)\s+(?:an?|the)\s+(?:[A-Z]?\w+\s+){0,2}"
        r"(?:director|actor|actress|singer|songwriter|composer|writer|musician|novelist|poet|painter|sculptor|philosopher|scientist|player|footballer|playwright|screenwriter|architect|engineer|comedian|politician|drummer|guitarist|bassist|vocalist|pianist|dramatist|journalist|biographer|historian|founder|inventor|professor|band|group|series|film|movie|novel|song|album|book|company|club|team|institution|organization|orchestra)\b"
        # 'X Y created/wrote/composed/...'
        r"|\b[A-Z][a-z]+(?:\s+[A-Z][a-z'\-]+){1,3}\s+(?:created|wrote|composed|directed|founded|invented|developed|produced|hosted|painted|sculpted|designed|established|discovered|won|earned)\b"

        # === MEMORY_RECALL TIER D: gpt-oss-style hedged self-recall ===
        # 'Actually X was Y...?'  (uncertainty about model's own memory)
        r"|(?i:\bactually\s+)[A-Z][\w'\-]+(?:\s+[\w'\-]+){1,6}\s+(?:was|is|did|wrote|played|composed|directed|founded|created|invented|won|born)\s+[^.!?]{0,40}\?"
        r")"
    ),

    # NEW: merged 'search_failure' (retrieval returned nothing useful) and
    # 'evidence_incomplete' (retrieval returned something but insufficient).
    # Both signal "the retrieval pipeline did not give what I need" and
    # both drive the same downstream behavior (query_rewrite or internal_use).
    'retrieval_inadequate': re.compile(
        r"(?:"
        # === SEARCH_FAILURE markers ===
        r"\bno results?(?:\s+(?:returned|found|appeared))?\b|"
        r"\bsearch (?:didn'?t|did not) (?:return|yield|show|find|give)\b|"
        r"\bsearch results? (?:didn'?t|did not|aren'?t|are not|weren'?t|were not) (?:specific|helpful|useful|relevant|sufficient|return|yield|specific enough)\b|"
        r"\bsearch (?:yielded|returned|gave) (?:nothing|no result|no useful|no specific|no direct)\b|"
        r"\bnot (?:very )?(?:helpful|specific|useful|relevant|conclusive)\b|"
        r"\bwrong page\b|"
        r"\bredirect(?:ing|ed)\s+(?:to|or)\b|"
        r"\bempty (?:extract|result|page|snippet|response)\b|"
        r"\bsearch result(?:s)? (?:isn'?t|is not|are not|aren'?t) (?:very )?(?:specific|helpful|useful|relevant|sufficient)\b|"
        # === EVIDENCE_INCOMPLETE markers ===
        r"\bdoesn'?t (?:include|show|mention|say|contain|specify|reveal|tell|have|cover|state)\b|"
        r"\bdoes not (?:include|show|mention|say|contain|specify|reveal|tell|have|cover|state)\b|"
        r"\bdidn'?t (?:include|show|mention|say|contain|specify|reveal|tell|have|cover|state)\b|"
        r"\bvery brief\b|"
        r"\b(?:extract|snippet|summary|page|result|description) (?:is|seems|appears|might be|may be) (?:limited|truncated|brief|short|incomplete|missing|sparse|very brief)\b|"
        r"\b(?:snippet|extract|summary|page|result|description) (?:doesn'?t|does not|didn'?t|did not) (?:include|show|mention|say|contain|specify|reveal|tell|have)\b|"
        r"\btruncated\b|"
        r"\bnot enough (?:information|details?|context|specificity|content)\b|"
        r"\b(?:might|may|could) not (?:include|contain|mention|specify|show)\b|"
        r"\b(?:only|just) (?:gives?|shows?|returns?|provides?) (?:a |the )?(?:snippet|extract|summary|brief|small)\b"
        r")",
        re.IGNORECASE,
    ),
}


DOMAIN_BY_BENCHMARK = {
    'math500': 'M',
    'gpqa_diamond': 'M',
    'hle': 'M',
    'hotpotqa': 'R',
    'musique': 'R',
}


def detect_events(text, domain):
    """
    Detect events in a reasoning trace.

    Args:
        text: raw reasoning_trajectory string.
        domain: 'M' or 'R'.

    Returns:
        List of dicts {'type', 'char_pos', 'char_end', 'match'},
        sorted by char_pos. Overlapping matches are deduplicated by
        keeping the longest match starting at the earliest position.
    """
    patterns = DOMAIN_M_PATTERNS if domain == 'M' else DOMAIN_R_PATTERNS

    raw = []
    for event_type, pattern in patterns.items():
        for m in pattern.finditer(text):
            raw.append({
                'type': event_type,
                'char_pos': m.start(),
                'char_end': m.end(),
                'match': m.group(0),
            })

    raw.sort(key=lambda e: (e['char_pos'], -(e['char_end'] - e['char_pos'])))

    deduped = []
    last_end = -1
    for e in raw:
        if e['char_pos'] >= last_end:
            deduped.append(e)
            last_end = e['char_end']

    return deduped


def char_to_token_pos(text, char_pos):
    """Approximate whitespace-token offset for a char position."""
    return len(text[:char_pos].split())
