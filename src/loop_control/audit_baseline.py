#!/usr/bin/env python3
"""CPU-only evidence mining. Findings are candidates, not causal training labels.

No model calls. Exact loops preserve numerals/operators. Answer-tail matches do
not imply a sufficient proof. Literal arithmetic checks do not infer semantics.
Recovery markers are positive recovery events, never negative error labels.
"""
import argparse
import ast
import bisect
import collections
import multiprocessing
import csv
import difflib
import hashlib
import html
import json
import math
import os
import re
import socket
import statistics
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

VERSION = 'waste-trace-audit-v1'
TOK = re.compile(r'\w+|[^\w\s]', re.UNICODE)
RECOVERY = re.compile(
    r'\b(?:I\s+(?:made\s+(?:a|an)\s+(?:mistake|error)|miscalculated|misread|forgot|overlooked)|'
    r'(?:that|this|my|the\s+previous)\s+(?:calculation|answer|result|assumption)?\s*'
    r'(?:is|was)\s+(?:wrong|incorrect)|'
    r'(?:there(?:\s+is|\x27s)|I\s+see)\s+(?:a|an)\s+(?:mistake|error)|'
    r'(?:let\s+me|we\s+(?:need|have)\s+to)\s+(?:correct|fix|redo)|'
    r'I\s+(?:used|wrote|counted)\b.{0,90}?\binstead\s+of)\b|'
    r'(?:я\s+(?:ошибся|ошиблась|забыл|перепутал)|допущена\s+ошибка|исправим\s+ошибку)',
    re.I)
ANSWER = re.compile(r'\b(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*', re.I)
EQ = re.compile(r'(?<![\w.\\^*/+\-])'
                r'(?P<lhs>-?\d+(?:\.\d+)?(?:\s*[*×÷/+^\-]\s*-?\d+(?:\.\d+)?){1,5})'
                r'\s*=\s*(?P<rhs>-?\d+(?:\.\d+)?)'
                r'(?!\w|\.\d)')


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def excerpt(text, lo, hi, context=180):
    return {'start_char': lo, 'end_char': hi,
            'quote': text[lo:min(hi, lo + 1400)],
            'context_before': text[max(0, lo-context):lo],
            'context_after': text[hi:hi+context]}


def exact_loops(text, tokens):
    """Find literal periodic regions, >=3 cycles and >=64 lexical units.

    Stable arithmetic with changing numbers cannot pass this test. Repeated
    statements may still be legitimate: epistemic usefulness requires review.
    """
    words = [m.group().casefold() for m in tokens]
    anchors = collections.defaultdict(list)
    found = []
    covered_until = -1
    for i in range(max(0, len(words)-8)):
        if i < covered_until:
            continue
        key = tuple(words[i:i+8])
        positions = anchors[key]
        for j in reversed(positions[-12:]):
            period = i-j
            if not 2 <= period <= 1024:
                continue
            if j < covered_until:
                continue
            end = i+8
            while end < len(words) and words[end] == words[j+(end-j) % period]:
                end += 1
            length = end-j
            if length >= max(64, period*3):
                lo, hi = tokens[j].start(), tokens[end-1].end()
                found.append({'category': 'loop', 'kind': 'exact_periodic_text',
                              'evidence_level': 'literal_repetition_verified',
                              'semantic_waste_confirmed': False,
                              'period_lexical_units': period,
                              'repetitions': round(length/period, 3),
                              'repeated_lexical_units': length-period,
                              'repeated_chars': hi-tokens[j+period].start(),
                              **excerpt(text, lo, hi)})
                covered_until = end
                break
        positions.append(i)
        if len(positions) > 16:
            del positions[:-16]
    return found


def sentences(text):
    # Paragraph/line boundaries preserve arithmetic; sentence cuts preserve offsets.
    out = []
    for m in re.finditer(r'[^\n]+', text):
        start = m.start()
        for part in re.finditer(r'.+?(?:[.!?](?=\s+[A-ZА-Я])|$)', m.group()):
            val = part.group().strip()
            if val:
                off = len(part.group())-len(part.group().lstrip())
                out.append((start+part.start()+off, start+part.end(), val))
    return out


def repeated_sentences(text):
    groups = collections.defaultdict(list)
    for lo, hi, value in sentences(text):
        norm = tuple(m.group().casefold() for m in TOK.finditer(value))
        if len(norm) >= 16 and hi-lo >= 70:
            groups[norm].append((lo, hi))
    out = []
    for norm, occurrences in groups.items():
        if len(occurrences) < 3:
            continue
        out.append({'category': 'loop', 'kind': 'recurrent_exact_statement',
                    'evidence_level': 'literal_repetition_verified',
                    'semantic_waste_confirmed': False,
                    'occurrences': [{'start_char': a, 'end_char': b} for a, b in occurrences],
                    'repetitions': len(occurrences),
                    'repeated_lexical_units': (len(occurrences)-1)*len(norm),
                    **excerpt(text, *occurrences[0])})
    return sorted(out, key=lambda x: x['repeated_lexical_units'], reverse=True)[:30]


def norm_answer(value):
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ''
    s = str(value).strip().casefold()
    s = re.sub(r'\\(?:text|mathrm|mathbf)\{([^{}]*)\}', r'\1', s)
    return re.sub(r'[\s$]', '', s).strip('.,;')


def explicit_answers(text):
    out = []
    for m in re.finditer(r'\\boxed\s*\{', text):
        depth, end = 1, m.end()
        while end < len(text) and depth and end-m.end() < 500:
            depth += (text[end] == '{')-(text[end] == '}')
            end += 1
        if depth == 0:
            out.append((m.start(), end, text[m.end():end-1], 'boxed'))
    for m in ANSWER.finditer(text):
        tail = text[m.end():m.end()+200]
        # A complete short answer, not an arbitrary substring/number inside a proof.
        first = re.split(r'[\n;]|\.(?!\d)', tail, maxsplit=1)[0]
        first = re.sub(r'\s+(?:because|since|which|and|so)\b.*', '', first, flags=re.I)
        if 0 < len(first.strip()) <= 100:
            out.append((m.start(), m.end()+len(first), first.strip(), 'answer_statement'))
    return sorted(out)


def answer_tails(trace, tokens):
    text = trace['text']
    refs = set()
    ref = norm_answer(trace.get('reference'))
    if ref:
        refs.add(ref)
    if trace.get('correct') is True and norm_answer(trace.get('answer')):
        refs.add(norm_answer(trace['answer']))
    if not refs:
        return []
    starts = [m.start() for m in tokens]
    out = []
    for lo, hi, answer, kind in explicit_answers(text):
        if norm_answer(answer) not in refs:
            continue
        tail = len(tokens)-bisect.bisect_left(starts, hi)
        if tail < 200:
            continue
        out.append({'category': 'overthinking',
                    'kind': 'correct_candidate_then_incorrect_final' if trace.get('correct') is False
                            else 'long_tail_after_matching_answer',
                    'evidence_level': 'answer_string_match_not_verified_sufficient_solution',
                    'semantic_waste_confirmed': False, 'answer': answer,
                    'answer_marker': kind, 'tail_lexical_units': tail,
                    'tail_chars': len(text)-hi, 'final_correct': trace.get('correct'),
                    **excerpt(text, lo, hi)})
        break
    return out


def safe_arithmetic(expr):
    expr = expr.replace('×', '*').replace('÷', '/').replace('^', '**')
    if len(expr) > 120:
        raise ValueError('too long')
    tree = ast.parse(expr, mode='eval')
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            if abs(n.value) > 10**12:
                raise ValueError('large constant')
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add): result = a+b
            elif isinstance(n.op, ast.Sub): result = a-b
            elif isinstance(n.op, ast.Mult): result = a*b
            elif isinstance(n.op, ast.Div): result = a/b
            elif isinstance(n.op, ast.Pow) and b.denominator == 1 and abs(b) <= 8:
                result = a**int(b)
            else: raise ValueError('unsupported arithmetic')
            if result.numerator.bit_length() > 256 or result.denominator.bit_length() > 256:
                raise ValueError('too large')
            return result
        raise ValueError('not numeric')
    return ev(tree)


def arithmetic_events(text):
    events = []
    for m in EQ.finditer(text):
        before = text[max(0, m.start()-100):m.start()]
        after = text[m.end():m.end()+20]
        # Exclude partial RHS, exponents, percent and equations inside TeX arguments.
        if re.match(r'\s*[*×÷/+^=\d%]|\s*-\s*\d', after):
            continue
        if re.search(r'\\(?:frac|sqrt|log)\s*\{?\s*$', before):
            continue
        if re.search(r'\b(?:suppose|assuming|assume|incorrect|wrong|not|modulo|mod|approximately|approx)\b', before, re.I):
            continue
        try:
            actual = safe_arithmetic(m['lhs'])
            stated = Fraction(m['rhs'])
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        # Rounded decimals are not classified as arithmetic errors.
        tol = Fraction(1, 10**max(0, len(m['rhs'].split('.')[1]))) if '.' in m['rhs'] else Fraction(0)
        valid = abs(actual-stated) <= tol/2
        events.append({'lhs_key': re.sub(r'\s+', '', m['lhs']).replace('×', '*').replace('÷', '/'),
                       'lhs': m['lhs'], 'stated': m['rhs'], 'computed': str(actual),
                       'valid_literal_equality': valid, **excerpt(text, m.start(), m.end())})
    return events


def recovery_events(trace, tokens):
    text = trace['text']
    markers = []
    for m in RECOVERY.finditer(text):
        markers.append({'category': 'recovery', 'kind': 'explicit_self_correction',
                        'evidence_level': 'self_report_only_error_not_independently_verified',
                        'recovery_is_not_penalized': True, 'error_origin_char': None,
                        **excerpt(text, m.start(), m.end(), 300)})
    eqs = arithmetic_events(text)
    starts = [m.start() for m in tokens]
    costly = []
    for i, bad in enumerate(eqs):
        if bad['valid_literal_equality']:
            continue
        good = next((x for x in eqs[i+1:] if x['lhs_key'] == bad['lhs_key'] and x['valid_literal_equality']), None)
        if not good:
            continue
        n = bisect.bisect_left(starts, good['end_char'])-bisect.bisect_left(starts, bad['start_char'])
        if n < 30:
            continue
        costly.append({'category': 'costly_error', 'kind': 'literal_arithmetic_error_then_correction',
                       'evidence_level': 'arithmetic_checked_context_and_dependency_require_review',
                       'semantic_waste_confirmed': False,
                       'error': bad, 'correction': good,
                       'intervening_lexical_units': n,
                       'intervening_work_all_error_dependent': False,
                       'prevention_cost': None, 'net_prevention_benefit': None,
                       'benefit_status': 'unknown_without_matched_counterfactual',
                       'recovery_is_not_penalized': True,
                       **excerpt(text, bad['start_char'], good['end_char'])})
    return markers[:50], eqs[:100], costly


def analyze_trace(t):
    assert t['role'] in ('TRAIN', 'DEV')
    text = t['text']
    toks = list(TOK.finditer(text))
    findings = exact_loops(text, toks)+repeated_sentences(text)+answer_tails(t, toks)
    markers, eqs, costly = recovery_events(t, toks)
    findings += costly
    sid = hashlib.sha256(t['trace_id'].encode()).hexdigest()[:20]
    for j, f in enumerate(findings):
        f.update(finding_id=f'{sid}:{j}', trace_id=t['trace_id'])
    return {'trace_id': t['trace_id'], 'source_scope': t['source_scope'],
            'solver': t['solver'], 'benchmark': t['benchmark'],
            'question_component': t['question_component'], 'question_id': t['question_id'],
            'role': t['role'], 'condition': t['condition'], 'end_status': t['end_status'],
            'correct': t['correct'], 'correctness_source': t['correctness_source'],
            'reasoning_tokens': t['reasoning_tokens'], 'token_unit': t['token_unit'],
            'lexical_units': len(toks), 'chars': len(text),
            'source_origin': t['source_origin'], 'raw_content_sha256': t['raw_content_sha256'],
            'findings': findings, 'recovery_markers': markers, 'arithmetic_events': eqs,
            'parent_id': t.get('parent_id')}


def condition_key(r):
    # Seed is the repeated draw; model/effort/cap/arm/temperature remain matched.
    return (r['source_scope'].split('/')[0], r['solver'], r['benchmark'],
            r['question_component'], json.dumps(r['condition'], sort_keys=True))


def peer_comparisons(results):
    groups = collections.defaultdict(list)
    for r in results:
        groups[condition_key(r)].append(r)
    comparisons = []
    for group, rr in groups.items():
        # Identical text copies are not independent successful demonstrations.
        unique = {r['raw_content_sha256']: r for r in rr}.values()
        good = [r for r in unique if r['correct'] is True and r['end_status'] == 'natural'
                and r['token_unit'] == 'native_reasoning_tokens' and (r['reasoning_tokens'] or 0)>0]
        if not good:
            continue
        shortest = min(good, key=lambda r: r['reasoning_tokens'])
        median = statistics.median(r['reasoning_tokens'] for r in good)
        for r in rr:
            if r['trace_id'] == shortest['trace_id'] or r['token_unit'] != 'native_reasoning_tokens':
                continue
            n = r['reasoning_tokens']
            if not isinstance(n, (int, float)) or n <= 0:
                continue
            ratio = n/shortest['reasoning_tokens']
            if ratio < 1.5 or n-shortest['reasoning_tokens'] < 256:
                continue
            if r['recovery_markers'] or any(f['category']=='costly_error' for f in r['findings']):
                kind='long_recovery_vs_short_correct_peer'
            elif r['correct'] is True:
                kind='long_correct_vs_short_correct_peer'
            elif r['correct'] is False:
                kind='long_incorrect_vs_short_correct_peer'
            else:
                continue
            comparisons.append({'kind':kind,'trace_id':r['trace_id'],
                                'peer_trace_id':shortest['trace_id'],
                                'solver':r['solver'],'benchmark':r['benchmark'],
                                'question_component':r['question_component'],
                                'long_correct':r['correct'],'long_end_status':r['end_status'],
                                'long_tokens':n,'short_tokens':shortest['reasoning_tokens'],
                                'observed_token_difference':n-shortest['reasoning_tokens'],
                                'ratio_to_shortest':ratio,'ratio_to_correct_median':n/median,
                                'correct_peer_count':len(good),'all_attempt_count':len(rr),
                                'recovery_marker_count':len(r['recovery_markers']),
                                'same_prefix_verified':False,'causal_saving':None,
                                'evidence_level':'matched_question_and_condition_observational_candidate',
                                'selection':'shortest observed correct peer; optimistic, not expected policy saving'})
    return sorted(comparisons,key=lambda r:r['observed_token_difference'],reverse=True)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--limit',type=int)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'trace_analysis.jsonl').exists():
        raise SystemExit('Refusing to overwrite an existing run; choose a new output directory.')
    start=time.time(); input_hash=digest(args.input)
    dump(out/'RUN_STATE.json',{'status':'RUNNING','started_utc':utc(),'pid':os.getpid(),'host':socket.gethostname(),
                             'input_sha256':input_hash,'workers':args.workers,'version':VERSION})
    def generate():
        with open(args.input) as f:
            for i,l in enumerate(f):
                if args.limit and i>=args.limit:break
                yield json.loads(l)
    results=[]; counts=collections.Counter(); bysrc=collections.defaultdict(collections.Counter)
    with multiprocessing.Pool(processes=args.workers) as pool, \
            (out/'trace_analysis.jsonl').open('w') as af, (out/'findings.jsonl').open('w') as ff:
        for r in pool.imap(analyze_trace,generate(),chunksize=8):
            results.append(r);af.write(json.dumps(r,ensure_ascii=False)+'\n')
            bysrc[r['source_scope']]['traces']+=1
            for f in r['findings']:
                counts[f['category']+'|'+f['kind']]+=1
                bysrc[r['source_scope']][f['category']+'_findings']+=1
                ff.write(json.dumps({**f,'solver':r['solver'],'benchmark':r['benchmark'],'role':r['role']},ensure_ascii=False)+'\n')
            if len(results)%1000==0:
                af.flush();ff.flush();print(json.dumps({'processed':len(results),'elapsed_seconds':round(time.time()-start)}),flush=True)
                dump(out/'RUN_STATE.json',{'status':'RUNNING','processed':len(results),'pid':os.getpid(),'host':socket.gethostname(),'updated_utc':utc()})
    peers=peer_comparisons(results);write_csv(out/'peer_candidates.csv',peers)
    with (out/'peer_candidates.jsonl').open('w') as f:
        for r in peers:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    category_traces={k:sum(any(f['category']==k for f in r['findings']) for r in results)
                     for k in ('loop','overthinking','costly_error')}
    summary={'status':'COMPLETE_CANDIDATE_MINING_NOT_CAUSAL_VALIDATION','created_utc':utc(),'version':VERSION,
             'host':socket.gethostname(),'workers':args.workers,'elapsed_seconds':time.time()-start,
             'input_sha256':input_hash,'code_sha256':digest(__file__),'analyzed_traces':len(results),
             'question_components':len({r['question_component'] for r in results}),
             'findings':dict(counts),'traces_by_category':category_traces,'peer_candidates':len(peers),
             'traces_with_explicit_recovery_markers':sum(bool(r['recovery_markers']) for r in results),
             'confirmed_profitable_prevention_cases':None,
             'confirmed_profitable_prevention_cases_note':'not evaluated; no matched stronger-span counterfactuals in this analysis',
             'sources':{k:dict(v) for k,v in bysrc.items()},
             'units':'full lengths: saved native reasoning tokens where available; spans: exact chars and lexical units, not native tokens',
             'limits':['Literal repetition is verified but its semantic usefulness is unjudged',
                       'Matching a correct answer before the end does not establish sufficient reasoning',
                       'Self-correction is not automatically a true error or useless work',
                       'Literal arithmetic needs context review; intervening work is not all error-dependent',
                       'Shortest correct peers do not estimate the causal gain of increasing span effort',
                       'No SFT/PO, no LLM inference, no protected TEST/HOLD payload']}
    dump(out/'SUMMARY.json',summary)
    write_csv(out/'source_coverage.csv',[{'source_scope':k,**v} for k,v in bysrc.items()])
    dump(out/'RUN_STATE.json',{'status':'COMPLETE','processed':len(results),'finished_utc':utc(),'host':socket.gethostname(),'pid':os.getpid()})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sources','limits')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
