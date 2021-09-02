from itertools import product
import numpy as np
import pytest

from collections import deque

def ex(a):
    return np.int32(a)

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



class Workload:
    def __init__(self):
        self.BLOCK_IN = 16
        self.BLOCK_OUT = 4

        self.W = 42
        self.H = 42
        self.C = 8
        self.S = 2

        self.TW = 14
        self.TH = 12
        self.TC = 4

        assert self.TC == self.BLOCK_OUT
        assert 9 <= self.BLOCK_IN

        self.ib = np.zeros((2*self.ib_sz,), dtype=np.int8)
        self.wb = np.zeros((2*self.wb_sz,), dtype=np.int8)
        self.ob = np.zeros((2*self.ob_sz,), dtype=np.int32)


    def wgt_sz(self):
        return self.BLOCK_IN * self.C


    def wi(self, c_outer, k0, k1, c_inner):
        return c_inner + self.TC * (k1 + 3 * k0 + self.BLOCK_IN * c_outer)


    def wi2(self, k0, k1, c):
        return self.wi(c // self.TC, k0, k1, c % self.TC)


    def inp_sz(self):
        return (self.W + 2) * (self.H + 2) * self.C


    def ii(self, i_outer, j_outer, c_outer, i_inner, j_inner, c_inner):
        return c_inner + self.TC * (j_inner + self.TW * j_outer + (self.W + 2) * (i_inner + self.TH * i_outer + (self.H + 2) * c_outer))


    def ii2(self, i, j, c):
        return self.ii(i // self.TH, j // self.TW, c // self.TC, i % self.TH, j % self.TW, c % self.TC)


    def out_sz(self):
        assert self.W % self.S == 0
        assert self.H % self.S == 0
        return self.W // self.S * self.H // self.S * self.C


    def oi(self, i_outer, j_outer, c_outer, i_inner, j_inner, c_inner):
        return c_inner + self.TC * (j_inner + self.TW * j_outer + self.W // self.S * (i_inner + self.TH * i_outer + self.H // self.S * c_outer))


    def oi2(self, i, j, c):
        return self.oi(i // self.TH, j // self.TW, c // self.TC, i % self.TH, j % self.TW, c % self.TC)

    def gold(self, inp, wgt):
        out = np.zeros((self.out_sz(),), dtype=np.int32)
        for i in range(0, self.H, self.S):
            for j in range(0, self.W, self.S):
                for c in range(self.C):
                    inner = 0
                    for k0, k1 in product(range(3), range(3)):
                        inner += ex(wgt[self.wi2(k0, k1, c)]) * ex(inp[self.ii2(i + k0, j + k1, c)])
                    out[self.oi2(i // self.S, j // self.S, c)] += inner
        return out


    def tiled(self, inp, wgt):
        assert self.C % self.TC == 0
        out = np.zeros((self.out_sz(),), dtype=np.int32)
        for c_outer in range(self.C // self.TC):
            for i_outer in range((self.H + self.TH - 1) // self.TH):
                for j_outer in range((self.W + self.TW - 1) // self.TW):
                    for i_inner in range(0, min(self.H - i_outer * self.TH, self.TH), self.S):
                        i = i_outer * self.TH + i_inner
                        for j_inner in range(0, min(self.W - j_outer * self.TW, self.TW), self.S):
                            j = j_outer * self.TW + j_inner
                            for c_inner in range(self.TC):
                                c = c_outer * self.TC + c_inner
                                inner = 0
                                for k0, k1 in product(range(3), range(3)):
                                    inner += ex(wgt[self.wi2(k0, k1, c)]) * ex(inp[self.ii2(i + k0, j + k1, c)])
                                out[self.oi2(i // self.S, j // self.S, c)] += inner
        return out


    @property
    def ib_lsz(self):
        return self.TC

    @property
    def ib_sz(self):
        return (self.TH + 2) * (self.TW + 2) * self.TC

    def ibi(self, i, j, c):
        return c + self.TC * (j + (self.TW + 2) * i)


    @property
    def wb_lsz(self):
        return self.BLOCK_IN * self.TC

    @property
    def wb_sz(self):
        return self.BLOCK_IN * self.TC


    def wbi(self, k0, k1, c):
        return c + self.TC * (k1 + 3 * k0)


    @property
    def ob_lsz(self):
        return self.TC

    @property
    def ob_sz(self):
        return self.TH // self.S * self.TW // self.S * self.TC


    def obi(self, i, j, c):
        return c + self.TC * (j + self.TW // self.S * i)


    def tiled_and_buffered(self, inp, wgt):
        assert self.C % self.TC == 0
        out = np.zeros((self.out_sz(),), dtype=np.int32)
        for c_outer in range(self.C // self.TC):
            # copy into wb
            for c_inner in range(self.TC):
                c = c_outer * self.TC + c_inner
                for k0, k1 in product(range(3), range(3)):
                    self.wb[self.wbi(k0, k1, c_inner)] = wgt[self.wi2(k0, k1, c)]

            for i_outer in range((self.H + self.TH - 1) // self.TH):
                for j_outer in range((self.W + self.TW - 1) // self.TW):
                    # copy into ib
                    for i_inner in range(0, min(self.H - i_outer * self.TH, self.TH) + 2):
                        i = i_outer * self.TH + i_inner
                        for j_inner in range(0, min(self.W - j_outer * self.TW, self.TW) + 2):
                            j = j_outer * self.TW + j_inner
                            for c_inner in range(self.TC):
                                c = c_outer * self.TC + c_inner
                                self.ib[self.ibi(i_inner, j_inner, c_inner)] = inp[self.ii2(i, j, c)]


                    for i_inner in range(0, min(self.H - i_outer * self.TH, self.TH), self.S):
                        for j_inner in range(0, min(self.W - j_outer * self.TW, self.TW), self.S):
                            for c_inner in range(self.TC):
                                inner = 0
                                for k0, k1 in product(range(3), range(3)):
                                    inner += ex(self.wb[self.wbi(k0, k1, c_inner)]) * ex(self.ib[self.ibi(i_inner + k0, j_inner + k1, c_inner)])
                                self.ob[self.obi(i_inner // self.S, j_inner // self.S, c_inner)] = inner

                    # copy from ob
                    for i_inner in range(0, min(self.H - i_outer * self.TH, self.TH), self.S):
                        i = i_outer * self.TH + i_inner
                        for j_inner in range(0, min(self.W - j_outer * self.TW, self.TW), self.S):
                            j = j_outer * self.TW + j_inner
                            for c_inner in range(self.TC):
                                c = c_outer * self.TC + c_inner
                                out[self.oi2(i // self.S, j // self.S, c)] = self.ob[self.obi(i_inner // self.S, j_inner // self.S, c_inner)]

        return out


    def depthwise_conv1(self, range0, range1,
                        inp_stride0, inp_stride1,
                        wgt_stride0, wgt_stride1,
                        out_stride0, out_stride1,
                        uop_codes, S):
        """S (convolution stride) should be a parameter"""
        def ibi(i, j, c, o):
            return o * self.BLOCK_OUT + c + inp_stride1 * j + inp_stride0 * i

        def wbi(k0, k1, c, o):
            return o * self.BLOCK_IN * self.BLOCK_OUT + c + wgt_stride1 * k1 + wgt_stride0 * k0

        def obi(i, j, c, o):
            return o * self.BLOCK_OUT + c + out_stride1 * j + out_stride0 * i

        for i_inner in range(range0):
            for j_inner in range(range1):
                for inp_off, wgt_off, out_off in uop_codes:
                    for c_inner in range(self.TC):
                        inner = 0
                        for k0, k1 in product(range(3), range(3)):
                            inner += ex(self.wb[wbi(k0, k1, c_inner, wgt_off)]) * ex(self.ib[ibi(i_inner + k0, j_inner + k1, c_inner, inp_off)])
                        if i_inner % S == 0 and j_inner % S == 0:
                            self.ob[obi(i_inner // S, j_inner // S, c_inner, out_off)] = inner

    def depthwise_conv2(self, range0, range1,
                        inp_stride0, inp_stride1,
                        wgt_stride0, wgt_stride1,
                        out_stride0, out_stride1,
                        uop_codes, S):
        """S (convolution stride) should be a parameter"""
        def ibi(i, j, c, o):
            return o * self.BLOCK_OUT + c + inp_stride1 * j + inp_stride0 * i

        def wbi(k0, k1, c, o):
            return o * self.BLOCK_IN * self.BLOCK_OUT + c + wgt_stride1 * k1 + wgt_stride0 * k0

        def obi(i, j, c, o):
            return o * self.BLOCK_OUT + c + out_stride1 * j + out_stride0 * i

        def gen_gemm():
            while True:
                sq, i_inner, j_inner, wgt_off, out_off = yield
                for c_inner in range(self.TC):
                    inner = 0
                    for k0, k1 in product(range(3), range(3)):
                        inner += ex(self.wb[wbi(k0, k1, c_inner, wgt_off)]) * ex(sq[k0,k1,c_inner])
                    if i_inner % S == 0 and j_inner % S == 0:
                        self.ob[obi(i_inner // S, j_inner // S, c_inner, out_off)] = inner

        sink = gen_gemm()
        next(sink)

        # im2col
        for i_inner in range(range0):
            for j_inner in range(range1):
                for inp_off, wgt_off, out_off in uop_codes:
                    sq = np.zeros( (3,3,self.TC), dtype=np.int8)
                    for k0, k1 in product(range(3), range(3)):
                        for c_inner in range(self.TC):
                            sq[k0,k1,c_inner] = self.ib[ibi(i_inner + k0, j_inner + k1, c_inner, inp_off)]
                    sink.send( (sq, i_inner, j_inner, wgt_off, out_off))

    def depthwise_conv3(self, range0, range1,
                        inp_stride0, inp_stride1,
                        wgt_stride0, wgt_stride1,
                        out_stride0, out_stride1,
                        uop_codes, S):

        """S (convolution stride) should be a parameter"""
        def ibi(i, j, c, o):
            return o * self.BLOCK_OUT + c + inp_stride1 * j + inp_stride0 * i

        def wbi(k0, k1, c, o):
            return o * self.BLOCK_IN * self.BLOCK_OUT + c + wgt_stride1 * k1 + wgt_stride0 * k0

        def obi(i, j, c, o):
            return o * self.BLOCK_OUT + c + out_stride1 * j + out_stride0 * i

        def gen_gemm():
            while True:
                sq, i_inner, j_inner, wgt_off, out_off = yield
                for c_inner in range(self.TC):
                    inner = 0
                    for k0, k1 in product(range(3), range(3)):
                        inner += ex(self.wb[wbi(k0, k1, c_inner, wgt_off)]) * ex(sq[k0,k1,c_inner])
                    if i_inner % S == 0 and j_inner % S == 0:
                        self.ob[obi(i_inner // S, j_inner // S, c_inner, out_off)] = inner


        sink = gen_gemm()
        next(sink)

        lb = Linebuffer(range1+2)
        for i_inner in range(range0+2):
            for j_inner in range(range1+2):
                for inp_off, wgt_off, out_off in uop_codes:
                    lb.add( [self.ib[ibi(i_inner, j_inner, c_inner, inp_off)] for c_inner in range(self.TC)])
                    if lb.valid:
                        sq = np.zeros( (3,3,self.TC), dtype=np.int8)
                        for k0, k1 in product(range(3), range(3)):
                            sq[k0,k1,:] = lb.wins[k0][k1]
                        sink.send( (sq, i_inner-2, j_inner-2, wgt_off, out_off))

    def load_wgt_inst(self, c_off, c_max, wgt, buf_off):
        for c_inner in range(c_max):
            for k0, k1 in product(range(3), range(3)):
                self.wb[buf_off + self.wbi(k0, k1, c_inner)] = wgt[self.wi2(k0, k1, c_off + c_inner)]

    def load_inp_inst(self, i_off, j_off, c_off, i_max, j_max, c_max, inp, buf_off):
        for i_inner in range(i_max):
            for j_inner in range(j_max):
                for c_inner in range(c_max):
                    self.ib[buf_off + self.ibi(i_inner, j_inner, c_inner)] = \
                        inp[self.ii2(i_off + i_inner, j_off + j_inner, c_off + c_inner)]

    def store_inst(self, i_off, j_off, c_off, i_max, j_max, c_max, out, buf_off):
        for i_inner in range(i_max):
            for j_inner in range(j_max):
                for c_inner in range(c_max):
                    out[self.oi2(i_off + i_inner, j_off + j_inner, c_off + c_inner)] = \
                        self.ob[buf_off + self.obi(i_inner, j_inner, c_inner)]


    def call_instruction(self, inp, wgt, out, depthwise_inst, tbl):
        if tbl['nm'] == 'depthwise_inst':
            depthwise_inst(
                tbl['range0'], tbl['range1'],
                tbl['inp_stride0'], tbl['inp_stride1'],
                tbl['wgt_stride0'], tbl['wgt_stride1'],
                tbl['out_stride0'], tbl['out_stride1'],
                tbl['uop_codes'], tbl['S'])
        elif tbl['nm'] == 'load_wgt_inst':
            self.load_wgt_inst( tbl['c_off'], tbl['c_max'], wgt, tbl['buf_off']) 
        elif tbl['nm'] == 'load_inp_inst':
            self.load_inp_inst( tbl['i_off'], tbl['j_off'], tbl['c_off'],
                                tbl['i_max'], tbl['j_max'], tbl['c_max'],
                                inp, tbl['buf_off'])
        elif tbl['nm'] == 'store_inst':
            self.store_inst( tbl['i_off'], tbl['j_off'], tbl['c_off'],
                             tbl['i_max'], tbl['j_max'], tbl['c_max'],
                             out, tbl['buf_off'])
        else:
            assert False


    def tiled_and_buffered_mapped(self, inp, wgt, depthwise_inst):
        assert self.C % self.TC == 0
        assert self.TH % self.S == 0
        assert self.TW % self.S == 0

        inst_stream = []
        def ci( tbl):
            inst_stream.append(tbl)

        wgt_parity = 0
        inp_parity = 0
        for c_outer in range(self.C // self.TC):
            ci( {'nm': 'load_wgt_inst', 'c_off': c_outer*self.TC, 'c_max': self.TC, 'buf_off': wgt_parity*self.wb_sz})

            for i_outer in range((self.H + self.TH - 1) // self.TH):
                for j_outer in range((self.W + self.TW - 1) // self.TW):
                    ci({'nm': 'load_inp_inst', 'i_off': i_outer*self.TH, 'j_off': j_outer*self.TW, 'c_off': c_outer*self.TC,
                        'i_max': min(self.H - i_outer * self.TH, self.TH) + 2,
                        'j_max': min(self.W - j_outer * self.TW, self.TW) + 2,
                        'c_max': self.TC,
                        'buf_off': inp_parity*self.ib_sz})

                    ci({'nm': 'depthwise_inst',
                        'range0': min(self.H - i_outer * self.TH, self.TH),
                        'range1': min(self.W - j_outer * self.TW, self.TW),
                        'inp_stride0': self.TC * (self.TW + 2), 'inp_stride1': self.TC,
                        'wgt_stride0': self.TC * 3, 'wgt_stride1': self.TC,
                        'out_stride0': self.TW // self.S * self.TC, 'out_stride1': self.TC,
                        'uop_codes': [(inp_parity*self.ib_sz//self.ib_lsz,
                                       wgt_parity*self.wb_sz//self.wb_lsz,
                                       inp_parity*self.ob_sz//self.ob_lsz)],
                        'S': self.S
                        }
                       )

                    ci({'nm': 'store_inst', 'i_off': i_outer*self.TH//self.S, 'j_off': j_outer*self.TW//self.S, 'c_off': c_outer*self.TC,
                        'i_max': (min(self.H - i_outer * self.TH, self.TH) + self.S - 1) // self.S,
                        'j_max': (min(self.W - j_outer * self.TW, self.TW) + self.S - 1) // self.S,
                        'c_max': self.TC,
                        'buf_off': inp_parity*self.ob_sz})

                    inp_parity = (inp_parity+1) % 2

            wgt_parity = (wgt_parity + 1) % 2

        out = np.zeros((self.out_sz(),), dtype=np.int32)
        for tbl in inst_stream:
            print(tbl)
            self.call_instruction( inp, wgt, out, depthwise_inst, tbl)

        return out


@pytest.fixture
def random_test():
    wl = Workload()
    wgt = np.random.randint(-128, 128, size=(wl.wgt_sz(),), dtype=np.int8)
    inp = np.random.randint(-128, 128, size=(wl.inp_sz(),), dtype=np.int8)
    out_gold = wl.gold(inp, wgt)
    return wgt,inp,out_gold,wl

def test_A(random_test):
    wgt,inp,out_gold,wl = random_test
    out = wl.tiled(inp, wgt)
    assert (out_gold == out).all()

def test_B(random_test):
    wgt,inp,out_gold,wl = random_test
    out = wl.tiled_and_buffered(inp, wgt)
    assert (out_gold == out).all()

def test_C1(random_test):
    wgt,inp,out_gold,wl = random_test
    out = wl.tiled_and_buffered_mapped(inp, wgt, depthwise_inst=wl.depthwise_conv1)
    assert (out_gold == out).all()

def test_C2(random_test):
    wgt,inp,out_gold,wl = random_test
    out = wl.tiled_and_buffered_mapped(inp, wgt, depthwise_inst=wl.depthwise_conv2)
    assert (out_gold == out).all()

def test_C3(random_test):
    wgt,inp,out_gold,wl = random_test
    out = wl.tiled_and_buffered_mapped(inp, wgt, depthwise_inst=wl.depthwise_conv3)
    assert (out_gold == out).all()
