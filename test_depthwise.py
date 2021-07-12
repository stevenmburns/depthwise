from itertools import product
import numpy as np

BLOCK_IN = 16
BLOCK_OUT = 4

W = 42
H = 42
C = 8
S = 2

TW = 14
TH = 12
TC = 4

assert TC == BLOCK_OUT
assert 9 <= BLOCK_IN


def wgt_sz():
    return BLOCK_IN * C


def wi(c_outer, k0, k1, c_inner):
    return c_inner + TC * (k1 + 3 * k0 + BLOCK_IN * c_outer)


def wi2(k0, k1, c):
    return wi(c // TC, k0, k1, c % TC)


def inp_sz():
    return (W + 2) * (H + 2) * C


def ii(i_outer, j_outer, c_outer,
       i_inner, j_inner, c_inner):
    return c_inner + TC * (j_inner + TW * j_outer + (W + 2) * (i_inner + TH * i_outer + (H + 2) * c_outer))


def ii2(i, j, c):
    return ii(i // TH, j // TW, c // TC, i % TH, j % TW, c % TC)


def out_sz():
    assert W % S == 0
    assert H % S == 0
    return W // S * H // S * C


def oi(i_outer, j_outer, c_outer,
       i_inner, j_inner, c_inner):
    return c_inner + TC * (j_inner + TW * j_outer + W // S * (i_inner + TH * i_outer + H // S * c_outer))


def oi2(i, j, c):
    return oi(i // TH, j // TW, c // TC, i % TH, j % TW, c % TC)


def ex(a):
    return np.int32(a)


def gold(inp, wgt):
    out = np.zeros((out_sz(),), dtype=np.int32)
    for i in range(0, H, S):
        for j in range(0, W, S):
            for c in range(C):
                inner = 0
                for k0, k1 in product(range(3), range(3)):
                    inner += ex(wgt[wi2(k0, k1, c)]) * ex(inp[ii2(i + k0, j + k1, c)])
                out[oi2(i // S, j // S, c)] += inner
    return out


def tiled(inp, wgt):
    assert C % TC == 0
    out = np.zeros((out_sz(),), dtype=np.int32)
    for c_outer in range(C // TC):
        for i_outer in range((H + TH - 1) // TH):
            for j_outer in range((W + TW - 1) // TW):
                for i_inner in range(0, min(H - i_outer * TH, TH), S):
                    i = i_outer * TH + i_inner
                    for j_inner in range(0, min(W - j_outer * TW, TW), S):
                        j = j_outer * TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer * TC + c_inner
                            inner = 0
                            for k0, k1 in product(range(3), range(3)):
                                inner += ex(wgt[wi2(k0, k1, c)]) * ex(inp[ii2(i + k0, j + k1, c)])
                            out[oi2(i // S, j // S, c)] += inner
    return out


ib = np.zeros(((TH + 2) * (TW + 2) * TC,), dtype=np.int8)


def ibi(i, j, c):
    return c + TC * (j + (TW + 2) * i)


wb = np.zeros((BLOCK_IN * TC,), dtype=np.int8)


def wbi(k0, k1, c):
    return c + TC * (k1 + 3 * k0)


ob = np.zeros((TH // S * TW // S * TC,), dtype=np.int32)


def obi(i, j, c):
    return c + TC * (j + TW // S * i)


def tiled_and_buffered(inp, wgt):
    assert C % TC == 0
    out = np.zeros((out_sz(),), dtype=np.int32)
    for c_outer in range(C // TC):
        for i_outer in range((H + TH - 1) // TH):
            for j_outer in range((W + TW - 1) // TW):
                # copy into ib
                for i_inner in range(0, min(H - i_outer * TH, TH) + 2):
                    i = i_outer * TH + i_inner
                    for j_inner in range(0, min(W - j_outer * TW, TW) + 2):
                        j = j_outer * TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer * TC + c_inner
                            ib[ibi(i_inner, j_inner, c_inner)] = inp[ii2(i, j, c)]

                # copy into wb
                for c_inner in range(TC):
                    c = c_outer * TC + c_inner
                    for k0, k1 in product(range(3), range(3)):
                        wb[wbi(k0, k1, c_inner)] = wgt[wi2(k0, k1, c)]

                for i_inner in range(0, min(H - i_outer * TH, TH), S):
                    for j_inner in range(0, min(W - j_outer * TW, TW), S):
                        for c_inner in range(TC):
                            inner = 0
                            for k0, k1 in product(range(3), range(3)):
                                inner += ex(wb[wbi(k0, k1, c_inner)]) * ex(ib[ibi(i_inner + k0, j_inner + k1, c_inner)])
                            ob[obi(i_inner // S, j_inner // S, c_inner)] = inner

                # copy from ob
                for i_inner in range(0, min(H - i_outer * TH, TH), S):
                    i = i_outer * TH + i_inner
                    for j_inner in range(0, min(W - j_outer * TW, TW), S):
                        j = j_outer * TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer * TC + c_inner
                            out[oi2(i // S, j // S, c)] = ob[obi(i_inner // S, j_inner // S, c_inner)]

    return out


def depthwise_conv(range0, range1,
                   inp_stride0, inp_stride1,
                   wgt_stride0, wgt_stride1,
                   out_stride0, out_stride1,
                   uop_codes):
    """S (convolution stride) should be a parameter"""
    def ibi(i, j, c, o):
        return o * BLOCK_OUT + c + inp_stride1 * j + inp_stride0 * i

    def wbi(k0, k1, c, o):
        return o * BLOCK_IN * BLOCK_OUT + c + wgt_stride1 * k1 + wgt_stride0 * k0

    def obi(i, j, c, o):
        return o * BLOCK_OUT + c + out_stride1 * j + out_stride0 * i

    for i_inner in range(range0):
        for j_inner in range(range1):
            for inp_off, wgt_off, out_off in uop_codes:
                for c_inner in range(TC):
                    inner = 0
                    for k0, k1 in product(range(3), range(3)):
                        inner += ex(wb[wbi(k0, k1, c_inner, wgt_off)]) * ex(ib[ibi(i_inner + k0, j_inner + k1, c_inner, inp_off)])
                    if i_inner % S == 0 and j_inner % S == 0:
                        ob[obi(i_inner // S, j_inner // S, c_inner, out_off)] = inner


def tiled_and_buffered_mapped(inp, wgt):
    assert C % TC == 0
    out = np.zeros((out_sz(),), dtype=np.int32)
    for c_outer in range(C // TC):
        for i_outer in range((H + TH - 1) // TH):
            for j_outer in range((W + TW - 1) // TW):
                # Need to refactor into a load_inp instruction call
                for i_inner in range(0, min(H - i_outer * TH, TH) + 2):
                    i = i_outer * TH + i_inner
                    for j_inner in range(0, min(W - j_outer * TW, TW) + 2):
                        j = j_outer * TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer * TC + c_inner
                            ib[ibi(i_inner, j_inner, c_inner)] = inp[ii2(i, j, c)]
                # Need to refactor into a load_wgt instruction call
                for c_inner in range(TC):
                    c = c_outer * TC + c_inner
                    for k0, k1 in product(range(3), range(3)):
                        wb[wbi(k0, k1, c_inner)] = wgt[wi2(k0, k1, c)]

                depthwise_conv(min(H - i_outer * TH, TH), min(W - j_outer * TW, TW),
                               TC * (TW + 2), TC, TC * 3, TC, TW // S * TC, TC,
                               [(0, 0, 0)])

                # Need to refactor into a store_acc instruction call
                for i_inner in range(0, min(H - i_outer * TH, TH), S):
                    i = i_outer * TH + i_inner
                    for j_inner in range(0, min(W - j_outer * TW, TW), S):
                        j = j_outer * TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer * TC + c_inner
                            out[oi2(i // S, j // S, c)] = ob[obi(i_inner // S, j_inner // S, c_inner)]

    return out


def test_A():
    wgt = np.random.randint(0, 127, size=(wgt_sz(),), dtype=np.int8)
    inp = np.random.randint(0, 127, size=(inp_sz(),), dtype=np.int8)

    out_gold = gold(inp, wgt)
    out = tiled(inp, wgt)
    assert (out_gold == out).all()


def test_B():
    wgt = np.random.randint(0, 127, size=(wgt_sz(),), dtype=np.int8)
    inp = np.random.randint(0, 127, size=(inp_sz(),), dtype=np.int8)

    out_gold = gold(inp, wgt)
    out = tiled_and_buffered(inp, wgt)
    assert (out_gold == out).all()


def test_C():
    wgt = np.random.randint(0, 127, size=(wgt_sz(),), dtype=np.int8)
    inp = np.random.randint(0, 127, size=(inp_sz(),), dtype=np.int8)

    out_gold = gold(inp, wgt)
    out = tiled_and_buffered_mapped(inp, wgt)
    assert (out_gold == out).all()
