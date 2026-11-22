"""三角形条带化（triangle stripification）核心实现。

公开 API
--------
``TriStripper.build(triangles) -> list[list[int]]``
    输入三角形列表（每个三角形是 3 个非负整数顶点索引），
    返回若干条带；每条带为顶点索引列表，连续三个顶点构成一个
    输入三角形，所有输入三角形恰好被覆盖一次。
``TriStripper.coverage() -> int``
    最近一次 build 覆盖的三角形数。
``TriStripper.reuse_ratio() -> float``
    平均顶点复用率，定义为节省的顶点槽位占比
    ``(3T - 输出顶点总数) / (3T)``，取值区间 ``[0, 2/3]``。

贪心策略（确定性）
------------------
1. 以“未访问邻接三角形数”最少的三角形作为条带种子
   （懒删除最小堆选取，平局按三角形 id，保证确定性）；
2. 在种子的三条边中贪心选择邻接三角形最少的边作为**起始边**
   （边界边优先），并在剩余两条边中选邻接较多者作为条带另一端的
   延伸边，避免把唯一的连接边埋成条带对角线；
3. 从条带两端贪心延伸，每次选择当前未访问邻接数最少的邻接三角形，
   且条带内顶点不允许重复；所有平局按三角形 id / 边键打破。

并发
----
``build`` 的全部工作数据都是函数内局部变量，不同线程（无论是否
共享实例）不会互相污染计算；实例上的结果发布与统计读取由锁保护，
保证读到的是一致快照。
"""

from collections import deque
import heapq
import threading


def _edge_key(a, b):
    """无向边的确定性键。"""
    return (a, b) if a < b else (b, a)


def _third_vertex(a, b, c, key):
    """返回三角形 (a,b,c) 中不属于边 key 的那个顶点。"""
    p, q = key
    if a != p and a != q:
        return a
    if b != p and b != q:
        return b
    return c


class TriStripper:
    def __init__(self):
        self._lock = threading.RLock()
        self._strips = []
        self._coverage = 0
        self._reuse_ratio = 0.0

    # ------------------------------------------------------------------ API

    def build(self, triangles):
        """生成条带，返回 ``list[list[int]]``；结果同时保存在实例上。"""
        strips, coverage, reuse = self._build(triangles)
        with self._lock:
            self._strips = strips
            self._coverage = coverage
            self._reuse_ratio = reuse
        return strips

    def coverage(self):
        """覆盖三角形数。"""
        with self._lock:
            return self._coverage

    def reuse_ratio(self):
        """平均顶点复用率，取值 [0, 2/3]；未 build 过时为 0.0。"""
        with self._lock:
            return self._reuse_ratio

    # ------------------------------------------------------------- internals

    @staticmethod
    def _validate(triangles):
        normalized = []
        for tri in triangles:
            if not isinstance(tri, (tuple, list)) or len(tri) != 3:
                raise ValueError(
                    "每个三角形必须包含恰好 3 个顶点索引: %r" % (tri,)
                )
            a, b, c = tri
            for v in (a, b, c):
                if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                    raise ValueError(
                        "顶点索引必须是非负整数: %r" % (tri,)
                    )
            if a == b or b == c or a == c:
                raise ValueError(
                    "退化三角形（三个顶点必须互不相同）: %r" % (tri,)
                )
            normalized.append((a, b, c))
        return normalized

    def _build(self, triangles):
        tris = self._validate(triangles)
        t_count = len(tris)
        if t_count == 0:
            return [], 0, 0.0

        # 边 -> 共享该边的三角形 id 列表
        edge_tris = {}
        for tid, (a, b, c) in enumerate(tris):
            for key in (_edge_key(a, b), _edge_key(b, c), _edge_key(a, c)):
                bucket = edge_tris.get(key)
                if bucket is None:
                    edge_tris[key] = [tid]
                else:
                    bucket.append(tid)

        # deg[t] = t 当前未访问的（去重）邻接三角形数
        deg = [0] * t_count
        for tlist in edge_tris.values():
            if len(tlist) > 1:
                distinct = set(tlist)
                d = len(distinct) - 1
                for t in distinct:
                    deg[t] += d

        # 懒删除最小堆：按当前度数选种子三角形，平局按 id（确定性）
        heap = [(d, tid) for tid, d in enumerate(deg)]
        heapq.heapify(heap)
        visited = bytearray(t_count)
        strips = []

        def mark_visited(t):
            visited[t] = 1
            affected = set()
            a, b, c = tris[t]
            for key in (
                _edge_key(a, b),
                _edge_key(b, c),
                _edge_key(a, c),
            ):
                for u in edge_tris.get(key, ()):
                    if u != t and not visited[u]:
                        affected.add(u)
            for u in affected:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))

        def pop_seed():
            while heap:
                d, t = heapq.heappop(heap)
                if not visited[t] and d == deg[t]:
                    return t
            return -1

        def pick_neighbor(key, in_strip):
            """沿边 key 选最优未访问邻接三角形（度数最少，id 最小平局）。"""
            best = None
            best_w = -1
            for u in edge_tris.get(key, ()):
                if visited[u]:
                    continue
                w = _third_vertex(*tris[u], key)
                if w in in_strip:
                    # 保持条带内顶点互不相同
                    continue
                cand = (deg[u], u)
                if best is None or cand < best:
                    best = cand
                    best_w = w
            if best is None:
                return -1, -1
            return best[1], best_w

        def open_neighbor_count(key, seed):
            return sum(
                1 for u in edge_tris[key] if u != seed and not visited[u]
            )

        while True:
            seed = pop_seed()
            if seed < 0:
                break
            a, b, c = tris[seed]
            keys3 = (
                _edge_key(a, b),
                _edge_key(b, c),
                _edge_key(a, c),
            )
            # 贪心选起始边：未访问邻接最少（边界边优先），平局按键序
            start_key = min(
                keys3,
                key=lambda k: (open_neighbor_count(k, seed), k),
            )
            # 另一条暴露在条带边界上的边：邻接尽量多，把最“安静”的
            # 第三条边埋为条带对角线 (v0, v2)
            other_keys = [k for k in keys3 if k != start_key]
            back_key = max(
                other_keys,
                key=lambda k: (open_neighbor_count(k, seed), k),
            )
            # 两条边界边共享的顶点必须是条带中间顶点 v1：
            # v1=共享点, v0=起始边另一端, v2=后延伸边另一端
            shared = (
                start_key[0]
                if start_key[0] == back_key[0]
                or start_key[0] == back_key[1]
                else start_key[1]
            )
            p = start_key[1] if start_key[0] == shared else start_key[0]
            r = back_key[1] if back_key[0] == shared else back_key[0]
            q = shared
            mark_visited(seed)

            strip = deque((p, q, r))
            in_strip = {p, q, r}

            # 向前延伸（起始边一端，通常很快止于网格边界）
            while True:
                front_key = _edge_key(strip[0], strip[1])
                u, w = pick_neighbor(front_key, in_strip)
                if u < 0:
                    break
                mark_visited(u)
                strip.appendleft(w)
                in_strip.add(w)

            # 向后延伸（主延伸方向）
            while True:
                back = _edge_key(strip[-2], strip[-1])
                u, w = pick_neighbor(back, in_strip)
                if u < 0:
                    break
                mark_visited(u)
                strip.append(w)
                in_strip.add(w)

            strips.append(list(strip))

        total_vertices = sum(len(strip) for strip in strips)
        reuse = (3 * t_count - total_vertices) / (3 * t_count)
        return strips, t_count, reuse
