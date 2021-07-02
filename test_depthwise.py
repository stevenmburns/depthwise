
import numpy as np

W = 42
H = 42
C = 8
S = 2

TW = 14
TH = 12
TC = 4



def wgt_sz():
    return 3*3*C

def wgt_idx( c_outer, k0, k1, c_inner):
    return c_inner + TC*(k1 + 3*(k0 + 3*c_outer))

def wgt_idx2( k0, k1, c):
    return wgt_idx( c // TC, k0, k1, c %TC)

def inp_sz():
    return (W+2)*(H+2)*C

def inp_idx( i_outer, j_outer, c_outer,
             i_inner, j_inner, c_inner):
    return c_inner + TC*(j_inner + TW*j_outer + (W+2)*(i_inner + TH*i_outer + (H+2)*c_outer))

def inp_idx2( i, j, c):
    return inp_idx( i//TH, j//TW, c//TC, i%TH, j%TW, c%TC)

def out_sz():
    assert W % S == 0
    assert H % S == 0
    return W//S*H//S*C

def out_idx( i_outer, j_outer, c_outer,
             i_inner, j_inner, c_inner):
    return c_inner + TC*(j_inner + TW*j_outer + W//S*(i_inner + TH*i_outer + H//S*c_outer))

def out_idx2( i, j, c):
    return out_idx( i//TH, j//TW, c//TC, i%TH, j%TW, c%TC)

import itertools

def gold( inp, wgt):
    out = np.zeros( (out_sz(),), dtype=np.int32)
    for i in range(0,H,S):
        for j in range(0,W,S):
            for c in range(C):
                inner = 0
                for k0,k1 in itertools.product( range(0,3), range(0,3)):
                    inner += np.int32(wgt[wgt_idx2(k0,k1,c)]) * \
                             np.int32(inp[inp_idx2(i+k0,j+k1,c)])
                out[out_idx2(i//S,j//S,c)] += inner
    return out

def tiled( inp, wgt):
    assert C % TC == 0
    out = np.zeros( (out_sz(),), dtype=np.int32)
    for c_outer in range(C//TC):
        for i_outer in range((H+TH-1)//TH):
            for j_outer in range((W+TW-1)//TW):
                for i_inner in range(0,min(H-i_outer*TH,TH),S):
                    i = i_outer*TH + i_inner
                    for j_inner in range(0,min(W-j_outer*TW,TW),S):
                        j = j_outer*TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer*TC + c_inner
                            inner = 0
                            for k0,k1 in itertools.product( range(0,3), range(0,3)):
                                inner += np.int32(wgt[wgt_idx2(k0,k1,c)]) * \
                                         np.int32(inp[inp_idx2(i+k0,j+k1,c)])
                            out[out_idx2(i//S,j//S,c)] += inner
    return out

def tiled_and_buffered( inp, wgt):
    assert C % TC == 0
    out = np.zeros( (out_sz(),), dtype=np.int32)
    for c_outer in range(C//TC):
        for i_outer in range((H+TH-1)//TH):
            for j_outer in range((W+TW-1)//TW):
                
                inp_buff = np.zeros( ((TH+2)*(TW+2)*TC,), dtype=np.int8)
                def inp_buff_idx( i, j, c):
                    return c + TC*( j + (TW+2)*i)
                wgt_buff = np.zeros( (3*3*TC,), dtype=np.int8)
                def wgt_buff_idx( k0, k1, c):
                    return c + TC*( k1 + 3*k0)
                out_buff = np.zeros( (TH//S*TW//S*TC,), dtype=np.int32)
                def out_buff_idx( i, j, c):
                    return c + TC*( j + TW//S*i)
                
                # copy into inp_buff


                for i_inner in range(0,min(H-i_outer*TH,TH),S):                
                    i = i_outer*TH + i_inner
                    for j_inner in range(0,min(W-j_outer*TW,TW),S):
                        j = j_outer*TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer*TC + c_inner

                # copy into wgt_buff
                for c_inner in range(TC):
                    c = c_outer*TC + c_inner
                    for k0,k1 in itertools.product( range(0,3), range(0,3)):
                        wgt_buff[wgt_buff_idx(k0,k1,c_inner)] = wgt[wgt_idx2(k0,k1,c)]

                for i_inner in range(0,min(H-i_outer*TH,TH),S):
                    i = i_outer*TH + i_inner
                    for j_inner in range(0,min(W-j_outer*TW,TW),S):
                        j = j_outer*TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer*TC + c_inner
                            inner = 0
                            for k0,k1 in itertools.product( range(0,3), range(0,3)):
                                inner += np.int32(wgt_buff[wgt_buff_idx(k0,k1,c_inner)]) * \
                                         np.int32(inp[inp_idx2(i+k0,j+k1,c)])
                            out_buff[out_buff_idx(i_inner//S,j_inner//S,c_inner)] += inner


                # copy from out_buff

                for i_inner in range(0,min(H-i_outer*TH,TH),S):                
                    i = i_outer*TH + i_inner
                    for j_inner in range(0,min(W-j_outer*TW,TW),S):
                        j = j_outer*TW + j_inner
                        for c_inner in range(TC):
                            c = c_outer*TC + c_inner
                            out[out_idx2(i//S,j//S,c)] = out_buff[out_buff_idx(i_inner//S,j_inner//S,c_inner)]

    return out


def test_A():
    wgt = np.random.randint( 0, 127, size=(wgt_sz(),), dtype=np.int8)
    inp = np.random.randint( 0, 127, size=(inp_sz(),), dtype=np.int8)

    out_gold = gold(inp,wgt)
    out_tiled = tiled(inp,wgt)
    assert (out_gold == out_tiled).all()

def test_B():
    wgt = np.random.randint( 0, 127, size=(wgt_sz(),), dtype=np.int8)
    inp = np.random.randint( 0, 127, size=(inp_sz(),), dtype=np.int8)

    out_gold = gold(inp,wgt)
    out_tiled_and_buffered = tiled_and_buffered(inp,wgt)
    assert (out_gold == out_tiled_and_buffered).all()

    
