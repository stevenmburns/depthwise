from itertools import product
import numpy as np
import pytest

from functools import wraps

def logging_decorator(fn):
    @wraps(fn)
    def func(*a, **kw):
        print(f'{fn.__name__}({a}, {kw})')
        return fn(*a, *kw)
    return func


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


ib_lsz = TC
ib_sz = (TH + 2) * (TW + 2) * TC
ib = np.zeros((2*ib_sz,), dtype=np.int8)


def ibi(i, j, c):
    return c + TC * (j + (TW + 2) * i)


wb_lsz = BLOCK_IN * TC
wb_sz = BLOCK_IN * TC
wb = np.zeros((2*wb_sz,), dtype=np.int8)


def wbi(k0, k1, c):
    return c + TC * (k1 + 3 * k0)


ob_lsz = TC
ob_sz = TH // S * TW // S * TC

ob = np.zeros((2*ob_sz,), dtype=np.int32)


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

@logging_decorator
def depthwise_conv1(range0, range1,
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


from collections import deque

class Linebuffer:
    def __init__(self, width):
        self.width = width
        self.rows = [deque([None]*width), deque([None]*width)]
        self.wins = [deque([None]*3), deque([None]*3), deque([None]*3)]
        self.i = -1
        self.j = 0


    def add(self, el):
        for win in self.wins:
            win.popleft()

        x = self.rows[0].popleft()
        self.wins[0].append(x)

        x = self.rows[1].popleft()
        self.wins[1].append(x)
        self.rows[0].append(x)

        self.wins[2].append(el)
        self.rows[1].append(el)

        self.i += 1
        if self.i == self.width:
            self.j += 1
            self.i = 0

    @property
    def valid(self):
        return self.j >= 2 and self.i >= 2

    @property
    def sq(self):
        result = np.zeros( (3,3), dtype=np.int8)
        for k0, k1 in product(range(3), range(3)):
            result[k0,k1] = self.wins[k0][k1]
        return result

def test_linebuffer():
    lb = Linebuffer(7)
    count = 0
    for i,j in product(range(lb.width),range(lb.width)):
        lb.add( lb.width*i+j)
        if lb.valid:
            count += 1
    assert count == (lb.width-2)**2


@logging_decorator
def depthwise_conv2(range0, range1,
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

    def gen_gemm():
        while True:
            sq, i_inner, j_inner, wgt_off, out_off = yield
            for c_inner in range(TC):
                inner = 0
                for k0, k1 in product(range(3), range(3)):
                    inner += ex(wb[wbi(k0, k1, c_inner, wgt_off)]) * ex(sq[k0,k1,c_inner])
                if i_inner % S == 0 and j_inner % S == 0:
                    ob[obi(i_inner // S, j_inner // S, c_inner, out_off)] = inner

    sink = gen_gemm()
    next(sink)

    # im2col
    for i_inner in range(range0):
        for j_inner in range(range1):
            for inp_off, wgt_off, out_off in uop_codes:
                sq = np.zeros( (3,3,TC), dtype=np.int8)
                for k0, k1 in product(range(3), range(3)):
                    for c_inner in range(TC):
                        sq[k0,k1,c_inner] = ib[ibi(i_inner + k0, j_inner + k1, c_inner, inp_off)]
                sink.send( (sq, i_inner, j_inner, wgt_off, out_off))

@logging_decorator
def depthwise_conv3(range0, range1,
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

    def gen_gemm():
        while True:
            sq, i_inner, j_inner, wgt_off, out_off = yield
            for c_inner in range(TC):
                inner = 0
                for k0, k1 in product(range(3), range(3)):
                    inner += ex(wb[wbi(k0, k1, c_inner, wgt_off)]) * ex(sq[k0,k1,c_inner])
                if i_inner % S == 0 and j_inner % S == 0:
                    ob[obi(i_inner // S, j_inner // S, c_inner, out_off)] = inner


    sink = gen_gemm()
    next(sink)

    lb = Linebuffer(range1+2)
    for i_inner in range(range0+2):
        for j_inner in range(range1+2):
            for inp_off, wgt_off, out_off in uop_codes:
                lb.add( [ib[ibi(i_inner, j_inner, c_inner, inp_off)] for c_inner in range(TC)])
                if lb.valid:
                    sq = np.zeros( (3,3,TC), dtype=np.int8)
                    for k0, k1 in product(range(3), range(3)):
                        sq[k0,k1,:] = lb.wins[k0][k1]
                    sink.send( (sq, i_inner-2, j_inner-2, wgt_off, out_off))


@logging_decorator
def load_wgt_inst( c_off, c_max, wgt, buf_off):
    for c_inner in range(c_max):
        for k0, k1 in product(range(3), range(3)):
            wb[buf_off + wbi(k0, k1, c_inner)] = wgt[wi2(k0, k1, c_off + c_inner)]

@logging_decorator
def load_inp_inst( i_off, j_off, c_off, i_max, j_max, c_max, inp, buf_off):
    for i_inner in range(i_max):
        for j_inner in range(j_max):
            for c_inner in range(c_max):
                ib[buf_off + ibi(i_inner, j_inner, c_inner)] = \
                    inp[ii2(i_off + i_inner, j_off + j_inner, c_off + c_inner)]

@logging_decorator
def store_inst( i_off, j_off, c_off, i_max, j_max, c_max, out, buf_off):
    for i_inner in range(i_max):
        for j_inner in range(j_max):
            for c_inner in range(c_max):
                out[oi2(i_off + i_inner, j_off + j_inner, c_off + c_inner)] = \
                    ob[buf_off + obi(i_inner, j_inner, c_inner)]
    

def call_instruction( inp, wgt, out, depthwise_inst, tbl):
    if tbl['nm'] == 'depthwise_inst':
        depthwise_inst(
            tbl['range0'], tbl['range1'],
            tbl['inp_stride0'], tbl['inp_stride1'],
            tbl['wgt_stride0'], tbl['wgt_stride1'],
            tbl['out_stride0'], tbl['out_stride1'],
            tbl['uop_codes'])
    elif tbl['nm'] == 'load_wgt_inst':
        load_wgt_inst( tbl['c_off'], tbl['c_max'], wgt, tbl['buf_off']) 
    elif tbl['nm'] == 'load_inp_inst':
        load_inp_inst( tbl['i_off'], tbl['j_off'], tbl['c_off'],
                       tbl['i_max'], tbl['j_max'], tbl['c_max'],
                       inp, tbl['buf_off'])
    elif tbl['nm'] == 'store_inst':
        store_inst( tbl['i_off'], tbl['j_off'], tbl['c_off'],
                       tbl['i_max'], tbl['j_max'], tbl['c_max'],
                       out, tbl['buf_off'])
    else:
        assert False

def tiled_and_buffered_mapped(inp, wgt, depthwise_inst=depthwise_conv3):
    assert C % TC == 0
    assert TH % S == 0
    assert TW % S == 0
    out = np.zeros((out_sz(),), dtype=np.int32)

    wgt_parity = 0
    inp_parity = 0

    def ci( tbl):
        call_instruction( inp, wgt, out, depthwise_inst, tbl)

    for c_outer in range(C // TC):
        ci( {'nm': 'load_wgt_inst', 'c_off': c_outer*TC, 'c_max': TC, 'buf_off': wgt_parity*wb_sz})

        for i_outer in range((H + TH - 1) // TH):
            for j_outer in range((W + TW - 1) // TW):
                ci({'nm': 'load_inp_inst', 'i_off': i_outer*TH, 'j_off': j_outer*TW, 'c_off': c_outer*TC,
                    'i_max': min(H - i_outer * TH, TH) + 2,
                    'j_max': min(W - j_outer * TW, TW) + 2,
                    'c_max': TC,
                    'buf_off': inp_parity*ib_sz})

                ci({'nm': 'depthwise_inst',
                    'range0': min(H - i_outer * TH, TH),
                    'range1': min(W - j_outer * TW, TW),
                    'inp_stride0': TC * (TW + 2), 'inp_stride1': TC,
                    'wgt_stride0': TC * 3, 'wgt_stride1': TC,
                    'out_stride0': TW // S * TC, 'out_stride1': TC,
                    'uop_codes': [(inp_parity*ib_sz//ib_lsz,
                                   wgt_parity*wb_sz//wb_lsz,
                                   inp_parity*ob_sz//ob_lsz)]
                    }
                   )

                ci({'nm': 'store_inst', 'i_off': i_outer*TH//S, 'j_off': j_outer*TW//S, 'c_off': c_outer*TC,
                    'i_max': (min(H - i_outer * TH, TH) + S - 1) // S,
                    'j_max': (min(W - j_outer * TW, TW) + S - 1) // S,
                    'c_max': TC,
                    'buf_off': inp_parity*ob_sz})

                inp_parity = (inp_parity+1) % 2

        wgt_parity = (wgt_parity + 1) % 2
    return out


@pytest.fixture
def random_test():
    wgt = np.random.randint(-128, 128, size=(wgt_sz(),), dtype=np.int8)
    inp = np.random.randint(-128, 128, size=(inp_sz(),), dtype=np.int8)
    out_gold = gold(inp, wgt)
    return wgt,inp,out_gold

def test_A(random_test):
    wgt,inp,out_gold = random_test
    out = tiled(inp, wgt)
    assert (out_gold == out).all()

def test_B(random_test):
    wgt,inp,out_gold = random_test
    out = tiled_and_buffered(inp, wgt)
    assert (out_gold == out).all()

def test_C1(random_test):
    wgt,inp,out_gold = random_test
    out = tiled_and_buffered_mapped(inp, wgt, depthwise_inst=depthwise_conv1)
    assert (out_gold == out).all()

def test_C2(random_test):
    wgt,inp,out_gold = random_test
    out = tiled_and_buffered_mapped(inp, wgt, depthwise_inst=depthwise_conv2)
    assert (out_gold == out).all()

def test_C3(random_test):
    wgt,inp,out_gold = random_test
    out = tiled_and_buffered_mapped(inp, wgt, depthwise_inst=depthwise_conv3)
    assert (out_gold == out).all()
