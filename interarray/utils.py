# SPDX-License-Identifier: LGPL-2.1-or-later
# https://github.com/mdealencar/interarray

import inspect
from collections import namedtuple


def namedtuplify(namedtuple_typename='', **kwargs):
    NamedTuplified = namedtuple(namedtuple_typename,
                                tuple(str(kw) for kw in kwargs))
    return NamedTuplified(**kwargs)


class NodeTagger():
    # 50 digits, 'I' and 'l' were dropped
    alphabet = 'abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNOPQRSTUVWXYZ'
    value = {c: i for i, c in enumerate(alphabet)}

    def __getattr__(self, b50):
        dec = 0
        digit_value = 1
        if b50[0] < 'α':
            for digit in b50[::-1]:
                dec += self.value[digit]*digit_value
                digit_value *= 50
            return dec
        else:
            # for greek letters, only single digit is implemented
            return ord('α') - ord(b50[0]) - 1

    def __getitem__(self, dec):
        if dec is None:
            return '∅'
        elif isinstance(dec, str):
            return dec
        b50 = []
        if dec >= 0:
            while True:
                dec, digit = divmod(dec, 50)
                b50.append(self.alphabet[digit])
                if dec == 0:
                    break
            return ''.join(b50[::-1])
        else:
            return chr(ord('α') + (abs(dec) - 1) % 25)


F = NodeTagger()


class NodeStr():

    def __init__(self, fnT, T):
        self.fnT = fnT
        self.T = T

    def __call__(self, u, *args):
        nodes = tuple((self.fnT[n], n)
                      for n in (u,) + args if n is not None)
        out = '–'.join(F[n_] + ('' if n < self.T else f'({F[n]})')
                       for n_, n in nodes)
        if len(nodes) > 1:
            out = f'«{out}»'
        else:
            out = f'<{out}>'
        return out


class Alerter():

    def __init__(self, where, varname):
        self.where = where
        self.varname = varname
        self.f_creation = inspect.stack()[1].frame

    def __call__(self, text):
        i = self.f_creation.f_locals[self.varname]
        function = inspect.stack()[1].function
        if self.where(i, function):
            print(f'[{i}|{function}] ' + text)
