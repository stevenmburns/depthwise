
import numpy as np

W = 114
H = 114
C = 32
S = 1

TW = 40
TH = 30
TC = 16

wgt = np.random.randint( 0, 127, size=(3,3,C), dtype=np.int32)

inp = np.random.randint( 0, 127, size=(W+2,H+2,C), dtype=np.int32)

def gold( inp, wgt):
    out = np.zeros( (W,H,C), dtype=np.int32)
    for j in range(0,W,S):
        for i in range(0,H,S):
            for c in range(C):
                inner = 0
                for ii in range(0,3):
                    for jj in range(0,3):
                        inner += wgt[ii,jj,c] * inp[i+ii,j+jj,c]
                out[i//S,j//S,c] += inner
    return out

out = gold(inp,wgt)

print(out)

def tiled( inp, wgt):
    out = np.zeros( (W,H,C), dtype=np.int32)
    for j_outer in range(0,W+TW-1,TW):
        for i_outer in range(0,H+TH-1,TH):
            for c_outer in range(0,C,TC):
                for j_inner in range(0,min(W-j_outer,TW),S):
                    j = j_outer + j_inner
                    for i_inner in range(0,min(H-i_outer,TH),S):
                        i = i_outer + i_inner
                        for c_inner in range(TC):
                            c = c_outer + c_inner
                            inner = 0
                            for ii in range(0,3):
                                for jj in range(0,3):
                                    inner += wgt[ii,jj,c] * inp[i+ii,j+jj,c]
                            out[i//S,j//S,c] += inner
    return out


def test_A():
    out_gold = gold(inp,wgt)
    out_tiled = tiled(inp,wgt)
    assert (out_gold == out_tiled).all()

    
