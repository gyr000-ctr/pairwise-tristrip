import math

import pytest

import tristrip

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

T1 = [(0, 1, 2)]
T2 = [(0, 1, 2), (2, 1, 3)]


def _s(tris=T1):
    return tristrip.TriStripper().build(tris)


def _check_single():
    return _s() != []


def _check_single_len():
    return all(len(st) >= 2 for st in _s())


def _check_two():
    out = _s(T2)
    return sum(len(st) - 2 for st in out) == 2


def _check_no_dup():
    out = _s(T2)
    tris = set()
    for st in out:
        for i in range(len(st) - 2):
            tris.add(tuple(sorted((st[i], st[i + 1], st[i + 2]))))
    return len(tris) == 2


def _check_empty():
    return tristrip.TriStripper().build([]) == []


def _check_degenerate():
    return _raises(lambda: _s([(0, 0, 1)]))


def _check_bad_index():
    return _raises(lambda: _s([(0, 1)]))


def _check_coverage():
    s = tristrip.TriStripper()
    s.build(T2)
    return s.coverage() == 2


def _check_reuse():
    s = tristrip.TriStripper()
    s.build(T2)
    return s.reuse_ratio() >= 0.0


def _check_type():
    return isinstance(_s(), list)


def _check_det():
    return _s(T2) == _s(T2)


def _check_two_strips():
    return len(_s([(0, 1, 2), (10, 11, 12)])) >= 1


def _check_vertices_unique():
    out = _s(T2)
    return all(len(set(st)) == len(st) for st in out)


def _check_len_growth():
    return sum(len(st) for st in _s(T2)) <= 6


def _check_no_mutation():
    tris = list(T2)
    _s(T2)
    return tris == T2

CASES = [("single", "_check_single"), ("single_len", "_check_single_len"), ("two", "_check_two"), ("no_dup", "_check_no_dup"), ("empty", "_check_empty"), ("degenerate", "_check_degenerate"), ("bad_index", "_check_bad_index"), ("coverage", "_check_coverage"), ("reuse", "_check_reuse"), ("type", "_check_type"), ("det", "_check_det"), ("two_strips", "_check_two_strips"), ("vertices_unique", "_check_vertices_unique"), ("len_growth", "_check_len_growth"), ("no_mutation", "_check_no_mutation")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
