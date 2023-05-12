from bloom_filter2 import BloomFilter
import random
import string
import time
from collections import defaultdict
from bbhash_table import BBHashTable
import bbhash
import sys
import hashlib
from pympler import asizeof
import numpy as np 

def first_b(n: int, b: int):
    if b == 0:
        return 0
    mask = 1
    for i in range(b - 1):
        mask <<= 1
        mask |= 1
    return n & mask

def build_k(list_size: int, string_size: int) -> list:
    ks = []
    for i in range(list_size):
        ks.append(''.join(random.choices(string.ascii_lowercase, k=string_size)))

    return ks

def build_kprime(ks: list, string_size: int):
    ksprimes = []
    true_neg = 0
    
    for k in ks:
        rand_num = random.randint(0,9)
        if rand_num == 2:
            ksprimes.append(k)
        else:
            ksprimes.append(''.join(random.choices(string.ascii_lowercase, k=string_size)))
            true_neg += 1
    return ksprimes, true_neg


class MyBloomFilter:
    def __init__(self, size, error_rate) -> None:
        self.bloom = BloomFilter(max_elements=size, error_rate=error_rate)
    
    def build_bloom_filter(self, keys: list):
        for k in keys:
            self.bloom.add(k)

    def query_bloom(self, keys: list):
        bloom_query = []
        for k in keys:
            if (k in self.bloom):
                bloom_query.append(k)
            else:
                bloom_query.append(None)

        return bloom_query

    def get_false_positives(self, query_bloom, ks, kprime, true_neg):
        false_pos = 0
        false_neg = 0

        for i, k in enumerate(kprime):
            actually_in = k in ks
            in_bloom = query_bloom[i]
            if (actually_in and in_bloom is None): 
                false_neg += 1
            elif (not actually_in and in_bloom is not None ):
                false_pos += 1

        false_neg = false_neg/(false_neg + (len(ks) - true_neg))
        false_pos = false_pos/(false_pos + true_neg)
        print('the false neg in bloom is', false_neg)
        return false_pos


def bloom_filter_func(ks: list, error_rate: float, curr_k_list: int, kprime: list, true_neg: int):

    bloom = MyBloomFilter(len(ks), error_rate)

    bloom.build_bloom_filter(ks)

    start = time.time()

    bloom_query = bloom.query_bloom(kprime)

    end = time.time()

    runtime = end - start 

    false_pos = bloom.get_false_positives(bloom_query, ks, kprime, true_neg)

    print('time for bloom filter to run ks', curr_k_list, 'was', runtime, 'for error rate', error_rate)

    print('size of bloom filter for list', curr_k_list, 'in size', asizeof.asizeof(bloom.bloom))

    print('false positive for bloom filter', curr_k_list, 'is', false_pos)


class MyMPHF:
    def __init__(self, keys: list, num_threads, gamma):
        hashed_keys = []
        for k in keys:
            hashed_val = hash(k) % ((sys.maxsize + 1) * 2)
            hashed_keys.append(hashed_val)

        self.mph = bbhash.PyMPHF(hashed_keys, len(hashed_keys), num_threads, gamma)


    def query_MPHF(self, ksprime):
        query_bbhash = []

        for k in ksprime:
            hashed_val = hash(k) % ((sys.maxsize + 1) * 2)
            query_bbhash.append(self.mph.lookup(hashed_val))

        return query_bbhash
    
    def get_false_positives(self, query_bbhash, ks, kprime, true_neg):
        false_pos = 0
        false_neg = 0

        for i, k in enumerate(kprime):
            actually_in = k in ks
            in_bbhash = query_bbhash[i]
            if (actually_in and in_bbhash is None): 
                false_neg += 1
            elif (not actually_in and in_bbhash is not None ):
                false_pos += 1

        false_neg = false_neg/(false_neg + (len(ks) - true_neg))
        false_pos = false_pos/(false_pos + true_neg)
        print('the false neg is for MPHF', false_neg)
        return false_pos

    def create_fingerprint(self, b, ks: list):
        numpylst = np.empty(len(ks))
        fingerprint_vecs = np.array(numpylst)

        for i, k in enumerate(ks):
            hashed_val = hash(k) % ((sys.maxsize + 1) * 2)
            fingerprint_vecs[self.mph.lookup(i)] = first_b(hashed_val, b)  

        return fingerprint_vecs
    
    def false_positive_fingerprint(self, fingerprint, b, kprime, ks, curr_k, true_neg):
        false_pos = 0
        false_neg = 0

        for i, k in enumerate(kprime):
            actually_in = k in ks
            in_mphf = self.mph.lookup(i)
            hashed_val = hash(k) % ((sys.maxsize + 1) * 2)

            if not actually_in and in_mphf is not None and in_mphf < len(ks) and fingerprint[in_mphf] ==first_b(hashed_val, b):
                false_pos+=1

            elif actually_in and in_mphf is None:
                false_neg+=1

        
        false_neg = false_neg/(false_neg + (len(ks) - true_neg))
        print(false_neg)
        false_pos = false_pos/(false_pos + true_neg)
        return false_pos
        



def testing_mphf(ks, ksprime, curr_k_list: int, true_neg: int):
    mph = MyMPHF(ks, 1, 1.0)

    start = time.time()

    query_bbhash = mph.query_MPHF(ksprime)

    end = time.time()

    runtime =  end - start

    false_positive = mph.get_false_positives(query_bbhash, ks, ksprime, true_neg)

    print('time for MPHF to run ks', curr_k_list, 'was', runtime, 'for list', curr_k_list)

    print('size of MPHF for list', curr_k_list, 'in size', asizeof.asizeof(mph.mph))

    print('false positive for MPHF', curr_k_list, 'is', false_positive)


    start = time.time()
    fingerprint_vecs7 = mph.create_fingerprint(7, ks)
    end = time.time()

    runtime1 =  end - start

    start = time.time()
    fingerprint_vecs8 = mph.create_fingerprint(8, ks)

    end = time.time()

    runtime2 =  end - start

    start = time.time()
    fingerprint_vecs10 = mph.create_fingerprint(10, ks)
    end = time.time()

    runtime3 =  end - start

    print('k', curr_k_list, 'false positive rate of', mph.false_positive_fingerprint(fingerprint_vecs7, 7, ksprime, ks, curr_k_list, true_neg), 'for finger7')
    print('k', curr_k_list, 'false positive rate of', mph.false_positive_fingerprint(fingerprint_vecs8, 8, ksprime, ks, curr_k_list, true_neg), 'for finger8')
    print('k', curr_k_list, 'false positive rate of', mph.false_positive_fingerprint(fingerprint_vecs10, 10, ksprime, ks, curr_k_list, true_neg), 'for finger10')

    print('   ')
    print('the time to query fingerpoint 7', ' for k', curr_k_list, 'is', runtime1)
    print('the time to query fingerpoint 8', ' for k', curr_k_list, 'is', runtime2)
    print('the time to query fingerpoint 10', ' for k', curr_k_list, 'is', runtime3)

    print('   ')
    print('for', curr_k_list, ' and fingerprint 7, the array + mphf size was', (asizeof.asizeof(mph.mph) + asizeof.asizeof(fingerprint_vecs7)))
    print('for', curr_k_list, ' and fingerprint 8, the array + mphf size was', (asizeof.asizeof(mph.mph) + asizeof.asizeof(fingerprint_vecs8)))
    print('for', curr_k_list, ' and fingerprint 10, the array + mphf size was', (asizeof.asizeof(mph.mph) + asizeof.asizeof(fingerprint_vecs10)))
    


if __name__ == '__main__':

    ks1 = build_k(10000, 32)
    ks2 = build_k(15000, 32)
    ks3 = build_k(50000, 32)
    ks4 = build_k(20000, 32)

    kprime1, true_neg1 = build_kprime(ks1, 32)
    kprime2, true_neg2 = build_kprime(ks2, 32)
    kprime3, true_neg3 = build_kprime(ks3, 32)
    kprime4, true_neg4 = build_kprime(ks4, 32)

    curr_k_list = 1
    

    bloom_filter_func(ks1, 0.0078125, curr_k_list, kprime1, true_neg1)
    bloom_filter_func(ks1, 0.00390625, curr_k_list, kprime1, true_neg1)
    bloom_filter_func(ks1, 0.0009765625, curr_k_list, kprime1, true_neg1)
    testing_mphf(ks1, kprime1, curr_k_list, true_neg1)
    print("   ")
    print("   ")
    print("   ")
    curr_k_list += 1
    bloom_filter_func(ks2, 0.0078125, curr_k_list, kprime2, true_neg2)
    bloom_filter_func(ks2, 0.00390625, curr_k_list, kprime2, true_neg2)
    bloom_filter_func(ks2, 0.0009765625, curr_k_list, kprime2, true_neg2)
    testing_mphf(ks2, kprime2, curr_k_list, true_neg2)
    print("   ")
    print("   ")
    print("   ")
    curr_k_list += 1
    bloom_filter_func(ks3, 0.0078125, curr_k_list, kprime3, true_neg3)
    bloom_filter_func(ks3, 0.00390625, curr_k_list, kprime3, true_neg3)
    bloom_filter_func(ks3, 0.0009765625, curr_k_list, kprime3, true_neg3)
    testing_mphf(ks3, kprime3, curr_k_list, true_neg3)
    print("   ")
    print("   ")
    print("   ")
    curr_k_list += 1
    bloom_filter_func(ks4, 0.0078125, curr_k_list, kprime4, true_neg4)
    bloom_filter_func(ks4, 0.00390625, curr_k_list, kprime4, true_neg4)
    bloom_filter_func(ks4, 0.0009765625, curr_k_list, kprime4, true_neg4)
    testing_mphf(ks4, kprime4, curr_k_list, true_neg4)

