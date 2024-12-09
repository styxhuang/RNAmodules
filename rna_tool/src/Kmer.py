import itertools
from collections import Counter
import re

import pandas as pd

def Kmer(fastas, k, type="RNA_features", upto=False, normalize=True, **kw):
    def _kmerArray(sequence, k):
        kmer = []
        for i in range(len(sequence) - k + 1):
            kmer.append(sequence[i:i + k])
        return kmer

    encoding = []
    header = ['#', 'label']
    NA = 'ACGU'
    if type in ("RNA_features", 'DNA'):
        NA = 'ACGU'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = _kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))

        for i in fastas:
            sequence = i.strip()
            kmers = _kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    kmerdf = pd.DataFrame(encoding)
    return kmerdf