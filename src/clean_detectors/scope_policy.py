"""Explicit experimental scope; never silently drop data or change selection."""
import math


def fixed_final(config):
    policy = config.get('validation_policy', 'quarter_epoch_macro_ap')
    if policy not in ('quarter_epoch_macro_ap', 'fixed_final'):
        raise ValueError('Unknown validation policy')
    if policy == 'fixed_final':
        if config['epochs'] != 1 or config.get('validation_fractions') != []:
            raise ValueError('Budget pilot is exactly one epoch and no checkpoint search')
        if config.get('selective995'):
            raise ValueError('This pilot does not select a selective995 encoder')
        return True
    if 'validation_fractions' in config:
        raise ValueError('Legacy validation schedule cannot be overridden implicitly')
    return False


def validation_steps(config, steps_per_epoch):
    return set() if fixed_final(config) else {
        math.ceil(steps_per_epoch * x) for x in (.25, .5, .75, 1.)}
