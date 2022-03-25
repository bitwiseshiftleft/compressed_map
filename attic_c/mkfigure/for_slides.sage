rows = 125
cols = 144
beta = 16
aug = 8
hi = int(rows*0.4)

set_random_seed(0)
def pgmify(m,filename,malt=None):
    r,c = m.dimensions()
    g = 7
    with open(filename,"w") as f:
        f.write("P2 %d %d %d\n" % (c,r,g))
        for i in range(r):
            alt_seen = False
            m_seen = False
            for j in range(c):
                level = g - g*int(m[i,j])
                if malt is not None:
                    alt_seen = alt_seen or malt[i,j]
                    m_seen   = m_seen   or m[i,j]
                if alt_seen and not m_seen:
                    level = g-1
                f.write("%d " % level)
            f.write("\n")

def ppm_highlight(m,filename,hi=None,pad=1,inner=4,outer=1,aug=aug,hspread=True):
    r,c = m.dimensions()
    if hi is None: hi=int(r*0.4)
    g = 7
    ser = [[(g,g,g)]*(c+2*(inner+outer+pad)) for _ in range(r)]
    for i in range(r):
        for j in range(c):
            h = g//2 + (g-g//2)*int(1-m[i,j])
            ser[i][j+pad+inner+outer] = (h,h,h)
    
    # white inner border
    for i in range(inner+pad+1):
        for j in range(outer,c+2*inner+2*pad):
            ser[hi+i][j] = ser[hi-i][j] = (g,g,g)

    # red outer border
    for i in range(inner+pad+1,inner+pad+outer+1):
        for j in range(0,len(ser[0])):
            ser[hi+i][j] = ser[hi-i][j] = (g,0,0)
    for i in range(-inner-pad-outer,inner+pad+outer+1):
        for j in range(outer):
            ser[hi+i][j] = ser[hi+i][len(ser[0])-1-j] = (g,0,0)

    for j in range(c):
        if m[hi,j]:
            for a in range(-inner,1+inner):
                rg = 0
                if j >= c-aug:
                    rg = 0
                    color = (0,2*g//3,g//3)
                elif hspread:
                    #rg = inner
                    color = (0,0,g)
                else:
                    rg = 0
                    color = (0,0,g)
                for b in range(-rg,1+rg):
                    if a^2 + b^2 <= (inner+0.3)^2:
                        ser[hi+a][j+b+inner+pad+outer] = color

    with open(filename,"w") as f:
        f.write("P3 %d %d %d\n" % (len(ser[0]),len(ser),g))
        for row in ser:
            for color in row:
                f.write("%d %d %d " % color)
            f.write("\n")

def ref(m,aug=aug):
    r,c = m.dimensions()
    e = 0
    for j in range(c-aug):
        for i in range(e,r):
            if m[i,j]:
                m.swap_rows(i,e)
                for ii in range(e+1,r):
                    for jj in range(j+1,c):
                        m[ii,jj] += m[ii,j]*m[e,jj]
                    m[ii,j] = 0
                e += 1
                break
    return m

def peel(m):
    # obviously not the real peeling algorithm: n^3 and returns the same result
    r,c = m.dimensions()
    e = 0
    while e<r and e<c:
        for j in range(e,c):
            seen = None
            for i in range(e,r):
                if m[i,j]:
                    if seen is not None:
                        seen = None # can't peel
                        break
                    else: seen = i
            if seen is not None:
                m.swap_rows(seen,e)
                m.swap_columns(j,e)
                break
        e += 1
    return m

def ribbon_matrix(rows,cols,beta,aug):
    m = Matrix(GF(2),rows,cols+aug)
    for i in range(rows):
        start = randint(0,cols-beta)
        m[i,start]=1
        for j in range(beta):
            m[i,start+1+j] = randint(0,1)
        for k in range(aug):
            m[i,cols+k] = randint(0,1)
    return m

def hamming_matrix(rows,cols,wt,aug):
    m = Matrix(GF(2),rows,cols+aug)
    for i in range(rows):
        w = 0
        while w < wt:
            j = randint(0,cols-1)
            if m[i,j] == 0: w+=1
            m[i,j] = 1
        for k in range(aug):
            m[i,cols+k] = randint(0,1)
    return m

def frayed_ribbon_matrix(rows,cols,beta,aug,k=1,force_stride=(hi,2)):
    beta = beta//2
    cols += (-cols) % beta
    blocks = cols//beta
    m = Matrix(GF(2),rows,cols+aug)
    k *= ln(blocks) / beta
    for i in range(rows):
        a = randint(0,blocks-1)
        stride = int(k/max(1e-6,random()^2))
        if i==force_stride[0]: stride=force_stride[1] # for the diagram
        b = (a+stride) % blocks
        if a==b: b = (b+1) % blocks
        for j in range(beta):
            m[i,a*beta+j] = randint(0,1)
            m[i,b*beta+j] = randint(0,1)
        for j in range(aug):
            m[i,cols+j] = randint(0,1)
    return m
    

m = ribbon_matrix(rows,cols,beta,aug)
ppm_highlight(m,"figure/ribbon_highlight.ppm",hspread=False)
pgmify(m,"figure/ribbon_unsorted.pgm")
m = Matrix(GF(2), rows,cols+aug, list(sorted(m.rows(),key=lambda r:str(r),reverse=True)))
pgmify(m,"figure/ribbon_sorted.pgm")
mm = ref(copy(m))
pgmify(mm,"figure/ribbon_ref.pgm")

m = hamming_matrix(rows,cols,5,aug)
pgmify(m,"figure/hamming_unsorted.pgm")
ppm_highlight(m,"figure/hamming_highlight.ppm")
mm = ref(copy(m))
pgmify(mm,"figure/hamming_ref.pgm")

m = hamming_matrix(rows,int(rows*1.3),3,aug)
pgmify(m,"figure/peel_unsorted.pgm")
ppm_highlight(m,"figure/peel_hilight.ppm")
mm = peel(copy(m))
pgmify(mm,"figure/peel_ref.pgm")

def frayed_blocks(row,beta,aug):
    beta = beta//2
    blocks = (len(row)-aug)//beta
    s = set()
    for i in range(blocks):
        for j in range(beta):
            if row[i*beta+j]:
                s.add(i)
                break
    assert len(s) == 2
    a,b = s
    return (min(a,b),max(a,b),blocks)

def frayed_key(beta,aug):
    def fn(row):
        a,b,blocks = frayed_blocks(row,beta,aug)
        if (a+blocks-b) < (b-a): a,b = b,a
        return a,-((b-a)%blocks)
    return fn

def frayed_key2(beta,aug):
    def fn(row):
        a,b,blocks = frayed_blocks(row,beta,aug)
        if (a+blocks-b) < (b-a): a,b = b,a
        xorstride = int(floor(log(a^^b,1.999)))
        ret =  xorstride,min(a,b)//(2^xorstride),a,b
        return ret
    return fn



def ref_frayed(m,m_orig,beta=beta,aug=aug,stage=0,red=False):
    r,c = m.dimensions()
    inscope = set([i for i in range(r)
            for (a,b,_) in [frayed_blocks(m_orig[i],beta,aug)]
            if a^^b <= 2^stage])
    inscope_max = max(*inscope)
    inscope_prev = set([i for i in range(r)
            for (a,b,_) in [frayed_blocks(m_orig[i],beta,aug)]
            if a^^b <= 2^(stage-1)])
    e = inscope_min = max(*(list(inscope_prev)+[-1,-1]))+1
    for j in range(c-aug):
        for i in range(e,inscope_max+1):
            if m[i,j]:
                m.swap_rows(i,e)
                for ii in range(e+1,r if red else inscope_max+1):
                    for jj in range(j+1,c):
                        m[ii,jj] += m[ii,j]*m[e,jj]
                    m[ii,j] = 0
                e += 1
                break

    def not_in_echelon(cc):
        if cc >= c-aug: return True,cc
        for i in range(inscope_max,-1,-1):
            if m[i,cc]:
                for j in range(cc):
                    if m[i,j]: return True,cc # not in echelon
                return False,cc # in echelon
        return True,cc

    # if red:
    #     perm = list(sorted(list(range(c)),key=not_in_echelon))
    #     m.permute_columns(Permutation([i+1 for i in perm]))

    return m

cols = rows
m = frayed_ribbon_matrix(rows,cols,beta,aug)
pgmify(m,"figure/frayed_unsorted.pgm")
ppm_highlight(m,"figure/frayed_hilight.ppm")
m = Matrix(GF(2), rows, m.dimensions()[1], list(sorted(m.rows(),key=frayed_key(beta,aug))))
pgmify(m,"figure/frayed_sorted.pgm")
m = Matrix(GF(2), rows, m.dimensions()[1], list(sorted(m.rows(),key=frayed_key2(beta,aug))))
pgmify(m,"figure/frayed_sorted_hier.pgm")
mm = copy(m)
for scope in range(int(ceil(log(ceil(cols/beta),2.00001)))+2):
    ref_frayed(mm,m,beta,aug,scope,False)
    pgmify(mm,"figure/frayed_sorted_ref_%d.0.pgm" % scope)
    ref_frayed(mm,m,beta,aug,scope,True)
    pgmify(mm,"figure/frayed_sorted_ref_%d.1.pgm" % scope)
