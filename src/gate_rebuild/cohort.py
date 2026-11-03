"""The frozen cohort of questions: one roster for every compared row (review findings GATE-V1-R4 and GATE-V2-R4).

The cohort is built from the FULL roster of the bank - decision states and attempts that ended before the decision position
alike - before any filter by row, and is written once to a file whose SHA256 the stages pin. A question enters the cohort only
when, for EVERY component of the chain, its role is known and is not a fit or selection role, and the alias audit by text found
nothing. Everything else is kept in the file with its reason and is out:
    dropped       the question is in a fit or selection role of a component;
    quarantined   its lineage or its alias status is unknown, or the alias audit found a suspect. Unknown is not clean.
All states of a question share one split. The same cohort serves every row of the matrix.

Version 2 of the format (GATE-V2-R4). The file holds the EXACT roster: every attempt with its key, question, split, whether it
reached the decision, and for a decision state the prefix hash of the bank. A bank whose attempts are not exactly these - another
key with the same counts, a missing or a repeated attempt, another role, another prefix - is refused. The file also holds the
SHA256 of the bank files it was built from, and a stage refuses bank files with another hash instead of only recording it.
A cohort may SUPPLEMENT an earlier cohort file: it names that file by SHA256 and states that the questions which are in are the
same. What was bound to the earlier file (the panel of stage S1) then stays valid without being rebuilt.

    python -B -m gate_rebuild.cohort --trainval F --replay F --lineage F --alias-audit F [--supplements OLD_COHORT] --out NEW_FILE
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

FORMAT = 'gate_rebuild_cohort_v2'
BANK_FILES = ('trainval', 'replay')
COMPONENTS = ('detector', 'process_v2')                 # components with their own question split; the lexical tagger is covered by the alias audit of its families
FORBIDDEN_ROLES = ('train', 'validation')
KNOWN_ROLES = FORBIDDEN_ROLES + ('test', 'absent_by_exact_key')
ALIAS_CLEAN = 'no_alias_found'


def canonical_sha256(cohort):
    return hashlib.sha256(json.dumps(cohort, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def roster_of(active, early):
    """Every attempt of the bank: key, question, split, whether it reached the decision, and the prefix hash of a decision state."""
    return [{'key': e.key, 'group': e.group, 'split': e.split, 'active': True, 'prefix_hash': extra['prefix_hash']} for e, extra in active] + \
           [{'key': e['key'], 'group': e['group'], 'split': e['split'], 'active': False, 'prefix_hash': None} for e in early]


def build_cohort(roster, lineage, aliases, components=COMPONENTS, inputs=None, supplements=None):
    """roster: every attempt of the bank; lineage: {group: {component: role}}; aliases: {group: {'status': ...}} of the alias audit."""
    if not roster or len({r['key'] for r in roster}) != len(roster):
        raise ValueError('the roster is empty or repeats an attempt')
    questions = {}
    for attempt in roster:
        entry = questions.setdefault(attempt['group'], {'split': attempt['split'], 'attempts': 0, 'decision_states': 0})
        if entry['split'] != attempt['split']:
            raise ValueError('question %s has attempts in two splits' % attempt['group'])
        entry['attempts'] += 1
        entry['decision_states'] += bool(attempt['active'])
    counts = collections.Counter()
    for group, entry in questions.items():
        roles = lineage.get(group)
        alias = (aliases.get(group) or {}).get('status')
        if roles is None or any(roles.get(c) not in KNOWN_ROLES for c in components):
            status = 'quarantined: lineage unknown'
        elif any(roles[c] in FORBIDDEN_ROLES for c in components):
            status = 'dropped: ' + ', '.join('%s %s' % (c, roles[c]) for c in components if roles[c] in FORBIDDEN_ROLES)
        elif alias is None:
            status = 'quarantined: alias audit missing'
        elif alias != ALIAS_CLEAN:
            status = 'quarantined: alias suspect'
        else:
            status = 'in'
        entry['status'] = status
        counts[(entry['split'], status.split(':')[0])] += 1
    cohort = {'format': FORMAT, 'components': list(components), 'forbidden_roles': list(FORBIDDEN_ROLES), 'inputs_sha256': dict(inputs or {}),
              'counts': [{'split': s, 'status': k, 'questions': n} for (s, k), n in sorted(counts.items())], 'questions': dict(sorted(questions.items())),
              'roster': sorted(({k: attempt[k] for k in ('key', 'group', 'split', 'active', 'prefix_hash')} for attempt in roster), key=lambda attempt: attempt['key']),
              'supplements': None}
    if supplements is not None:                                   # an earlier cohort file: the questions that are in must be the same
        earlier, earlier_sha = supplements
        inside = lambda value: sorted(g for g, e in value['questions'].items() if e['status'] == 'in')
        if inside(earlier) != inside(cohort) or any(earlier['questions'][g]['split'] != cohort['questions'][g]['split'] for g in inside(cohort)):
            raise ValueError('the cohort does not hold the same questions as the file it supplements')
        cohort['supplements'] = {'cohort_sha256': earlier_sha, 'same_questions_in': len(inside(cohort))}
    return cohort


def accepted_cohort_digests(cohort, file_sha256):
    """SHA256 values under which something may have been bound to this cohort: its own file and the file it supplements."""
    return {file_sha256} | ({cohort['supplements']['cohort_sha256']} if cohort.get('supplements') else set())


def check_bank_files(cohort, files):
    """files: {'trainval': path, 'replay': path}. The bank files must be the ones the cohort was built from."""
    from .data import file_sha256
    for name in BANK_FILES:
        if cohort['inputs_sha256'].get(name) != file_sha256(files[name]):
            raise ValueError('the bank file %s is not the one the cohort was built from' % name)


def apply_cohort(cohort, active, early):
    """The attempts of the questions that are IN the cohort. The roster of the bank must be EXACTLY the roster of the cohort:
    the same attempts with the same question, split, role and prefix. The order of the bank does not matter."""
    if cohort.get('format') != FORMAT:
        raise ValueError('not a cohort file of this version')
    frozen = {attempt['key']: attempt for attempt in cohort['roster']}
    if len(frozen) != len(cohort['roster']):
        raise ValueError('the cohort repeats an attempt')
    seen = set()
    for attempt in roster_of(active, early):
        if attempt['key'] in seen:
            raise ValueError('the bank repeats the attempt %s' % attempt['key'])
        seen.add(attempt['key'])
        if frozen.get(attempt['key']) != attempt:
            raise ValueError('attempt %s is not in the roster the cohort was built from, or differs from it' % attempt['key'])
    if seen != set(frozen):
        raise ValueError('the bank lacks %d attempts of the cohort' % len(set(frozen) - seen))
    inside = {group for group, entry in cohort['questions'].items() if entry['status'] == 'in'}
    return [(e, extra) for e, extra in active if e.group in inside], [e for e in early if e['group'] in inside], \
        {'questions_in': len(inside), 'questions_out': len(cohort['questions']) - len(inside)}


def main(argv=None):
    from .cost import CostModel
    from .data import file_sha256, load_trainext
    p = argparse.ArgumentParser()
    for name in ('--trainval', '--replay', '--lineage', '--alias-audit', '--out'):
        p.add_argument(name, required=True)
    p.add_argument('--supplements', default=None)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        sys.exit('refusing to overwrite %s' % out)
    neutral = CostModel('roster only', 1.0, 0.0, 'kept', {}, {}, 0.0)             # the cohort reads identities, no cost
    active, early = load_trainext(args.trainval, args.replay, neutral)
    supplements = None if args.supplements is None else (json.loads(Path(args.supplements).read_text()), file_sha256(args.supplements))
    cohort = build_cohort(roster_of(active, early), json.loads(Path(args.lineage).read_text())['questions'], json.loads(Path(args.alias_audit).read_text())['questions'],
                          inputs={name: file_sha256(getattr(args, name)) for name in ('trainval', 'replay', 'lineage', 'alias_audit')}, supplements=supplements)
    out.write_text(json.dumps(cohort, ensure_ascii=False, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'file_sha256': file_sha256(out), 'counts': cohort['counts']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
