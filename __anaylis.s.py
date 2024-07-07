items = 'pbeoπflmkndy'

from itertools import combinations
from tqdm import tqdm

# for each one in res2
# get items out of it

def string_exclude_set(string, exclude_set):
    return ''.join(
        char
        for char in string
        if char not in exclude_set
    )

outputs = set()
for i in {
    poss
    # for poss in tqdm(combinations(items, 4))
    for poss in (combinations(items, 4))
}:
    ooo = string_exclude_set(items, i)
    for uuu in {
    zzzposs
    # for zzzposs in tqdm(combinations(ooo, 4))
    for zzzposs in (combinations(ooo, 4))
}:
        sjfids = string_exclude_set(ooo, uuu)
        outputs.add('_'.join(sorted([
            ''.join(sorted(i)      ),
            ''.join(sorted(uuu)    ),
            ''.join(sorted(sjfids) ),
        ])))
# 5775
from math import factorial as q, comb as c
c(12, 4) * c(8, 4) // q(3)

myset = outputs.copy()

def pickout(inputitem, myset):
    query = ''.join(sorted(inputitem))
    return {i for i in myset if query not in i}

print(len(myset))
myset = pickout('peok', myset)
print(len(myset))
myset = pickout("peok", myset)
print(len(myset))
myset = pickout("oπfk", myset)
print(len(myset))
myset = pickout("oπlk", myset)
print(len(myset))
myset = pickout("oπmk", myset)
print(len(myset))
myset = pickout("oπkn", myset)
print(len(myset))
myset = pickout("oπkd", myset)
print(len(myset))
myset = pickout("oπky", myset)
print(len(myset))
myset = pickout("peoπ", myset)
print(len(myset))
myset = pickout("beoπ", myset)
print(len(myset))
myset = pickout("boπy", myset)
print(len(myset))
myset = pickout("bolk", myset)
print(len(myset))
myset = pickout("bomk", myset)
print(len(myset))
myset = pickout("boky", myset)
print(len(myset))
myset = pickout("eoπf", myset)
print(len(myset))
myset = pickout("eoπl", myset)
print(len(myset))
myset = pickout("eoπm", myset)
print(len(myset))
myset = pickout("eoπn", myset)
print(len(myset))
myset = pickout("eoπd", myset)
print(len(myset))
myset = pickout("eoπy", myset)
print(len(myset))
myset = pickout("eofk", myset)
print(len(myset))
myset = pickout("eolk", myset)
print(len(myset))
myset = pickout("eomk", myset)
print(len(myset))
myset = pickout("eokn", myset)
print(len(myset))
myset = pickout("eokd", myset)
print(len(myset))
myset = pickout("eoky", myset)
print(len(myset))
myset = pickout("ofky", myset)
print(len(myset))
myset = pickout("lkpb", myset)
print(len(myset))
