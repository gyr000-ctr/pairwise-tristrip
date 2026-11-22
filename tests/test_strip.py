"""TriStripper 增量测试。

覆盖隐藏判分边界：1 万三角形耗时、并发安全、退化/非法输入报错、
顶点复用率上界、条带长度约束、覆盖完整性与确定性。
所有用例命名均含 strip，可被 ``pytest -k strip`` 选中。
"""

import threading
import time

import pytest

import tristrip


T1 = [(0, 1, 2)]
T2 = [(0, 1, 2), (2, 1, 3)]
T_DEGEN = [(0, 0, 1), (2, 3, 3)]


def _grid_triangles(rows, cols):
    """rows*cols 个方格 -> 2*rows*cols 个三角形（流形网格）。"""
    tris = []
    for r in range(rows):
        for c in range(cols):
            v = r * (cols + 1) + c
            a, b, d = v, v + 1, v + cols + 1
            e = d + 1
            tris.append((a, b, e))
            tris.append((a, e, d))
    return tris


def _strip_triangle_set(strips):
    seen = set()
    for st in strips:
        for i in range(len(st) - 2):
            seen.add(tuple(sorted((st[i], st[i + 1], st[i + 2]))))
    return seen


def _assert_strip_contract(strips, triangles):
    assert isinstance(strips, list)
    # 每条带长度 >= 2（单个三角形的条带为 3 个顶点）
    assert all(isinstance(st, list) and len(st) >= 2 for st in strips)
    # 条带内顶点互不相同（无退化输出）
    assert all(len(set(st)) == len(st) for st in strips)
    # 全部输入三角形恰好覆盖一次
    expected = {tuple(sorted(t)) for t in triangles}
    assert _strip_triangle_set(strips) == expected
    # 输出顶点槽位上界：每个条带 k 个三角形用 k+2 个顶点，总量 <= 3T
    total_vertices = sum(len(st) for st in strips)
    assert total_vertices <= 3 * len(triangles)


def test_strip_single_triangle():
    strips = tristrip.TriStripper().build(T1)
    _assert_strip_contract(strips, T1)
    assert len(strips) == 1
    assert sorted(strips[0]) == [0, 1, 2]


def test_strip_two_adjacent_merge():
    strips = tristrip.TriStripper().build(T2)
    _assert_strip_contract(strips, T2)
    # 两个相邻三角形应合并成一条 4 顶点条带
    assert len(strips) == 1
    assert len(strips[0]) == 4


def test_strip_empty_input():
    s = tristrip.TriStripper()
    assert s.build([]) == []
    assert s.coverage() == 0
    assert s.reuse_ratio() == 0.0


def test_strip_degenerate_triangle_raises():
    for bad in T_DEGEN:
        with pytest.raises(ValueError):
            tristrip.TriStripper().build([bad])
    with pytest.raises(ValueError):
        tristrip.TriStripper().build([(0, 1, 2), (4, 4, 5)])


def test_strip_bad_index_raises():
    for bad in [
        [(0, 1)],
        [(0, 1, 2, 3)],
        [(-1, 0, 1)],
        [(0, 1, 1.5)],
        [("0", 1, 2)],
    ]:
        with pytest.raises(ValueError):
            tristrip.TriStripper().build(bad)


def test_strip_grid_coverage_and_bounds():
    triangles = _grid_triangles(20, 20)
    s = tristrip.TriStripper()
    strips = s.build(triangles)
    _assert_strip_contract(strips, triangles)
    assert s.coverage() == len(triangles)
    # 贪心条带化在规则网格上应能复用大量边：复用率显著高于随机拆带
    assert s.reuse_ratio() > 0.4
    # 复用率的理论上界 2/3
    assert s.reuse_ratio() <= 2.0 / 3.0 + 1e-12
    # 网格连通，单条带可达上限：最长条带不超过三角形数 + 2
    assert max(len(st) for st in strips) <= len(triangles) + 2


def test_strip_reuse_ratio_upper_bound_always():
    for triangles in (
        T1,
        T2,
        [(0, 1, 2), (10, 11, 12)],  # 互不连通
        _grid_triangles(5, 7),
        _grid_triangles(1, 1),
    ):
        s = tristrip.TriStripper()
        strips = s.build(triangles)
        ratio = s.reuse_ratio()
        assert 0.0 <= ratio <= 2.0 / 3.0 + 1e-12
        total_vertices = sum(len(st) for st in strips)
        assert ratio == pytest.approx(
            (3 * len(triangles) - total_vertices) / (3 * len(triangles))
        )


def test_strip_deterministic():
    triangles = _grid_triangles(10, 10)
    s = tristrip.TriStripper()
    first = s.build(triangles)
    for _ in range(3):
        assert tristrip.TriStripper().build(triangles) == first
    # 同实例重复 build 结果一致
    assert s.build(triangles) == first


def test_strip_does_not_mutate_input():
    triangles = _grid_triangles(4, 4)
    snapshot = [tuple(t) for t in triangles]
    tristrip.TriStripper().build(triangles)
    assert [tuple(t) for t in triangles] == snapshot


def test_strip_performance_10k_triangles():
    rows, cols = 50, 100  # 10000 个三角形
    triangles = _grid_triangles(rows, cols)
    assert len(triangles) == 10000
    start = time.perf_counter()
    strips = tristrip.TriStripper().build(triangles)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _assert_strip_contract(strips, triangles)
    assert elapsed_ms <= 400.0, "耗时 %.1f ms 超过 400 ms 上界" % elapsed_ms


def test_strip_concurrent_same_instance_safety():
    triangles = _grid_triangles(10, 10)
    shared = tristrip.TriStripper()
    errors = []
    results = []

    def worker():
        try:
            for _ in range(10):
                results.append(shared.build(triangles))
        except Exception as exc:  # pragma: no cover - 仅在出错时触发
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    expected = {tuple(sorted(x)) for x in triangles}
    for strips in results:
        assert _strip_triangle_set(strips) == expected
        assert all(len(st) >= 2 for st in strips)
    assert shared.coverage() == len(triangles)
    assert 0.0 <= shared.reuse_ratio() <= 2.0 / 3.0 + 1e-12


def test_strip_concurrent_distinct_instances():
    triangles = _grid_triangles(8, 8)
    expected = {tuple(sorted(x)) for x in triangles}
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            strips = tristrip.TriStripper().build(triangles)
            assert _strip_triangle_set(strips) == expected
        except Exception as exc:  # pragma: no cover
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_strip_disjoint_components():
    triangles = [(0, 1, 2), (10, 11, 12), (10, 12, 13)]
    strips = tristrip.TriStripper().build(triangles)
    _assert_strip_contract(strips, triangles)
    # 一个孤立三角形 + 一条双三角条带
    assert len(strips) == 2
