Overall for this project make sure that you have Python downloaded on your machine. For a lot of the installs, it is important to have the latest version of Python, which at this time is 3.11.3

Bloom Filter 

This file imports a bloom filters library and uses it to create a bloom filter class. This class allows you to initialize a BloomFilter with a max number of elements and error rate, build the bloom filter, query it using a set of keys, and calcuate the percentages of false positives. It also tells you the amount of time it takes to do the query on the bloom filter for K' as well as the size of the bloom filter in memory. This file also includes functions called "build_k" and "build_kprime" which allow building a list of keys, intended for the bloomfilter, of random strings which are a set size. One major difference between these two functions is that "build_kprimes" requires an input of a list of keys (which can be created using build_k) and adds a pseudo-random number of these keys to the new kprime. To run all of these functions from the class together run the "bloom_filter_func". This will give you print lines with information about the size of the bloom filter, time to query, and false positives.


In order to use these functions you must make sure to install this library  
    - pip3 install bloom-filter2
    - pip3 install pympler
     
Then you can either update main or use the functions directly on the command line

Minimum Perfect Hashes Fucntion

This file imports a Minimum Perfect Hashes Fucntion Library. The class created called MyMPHF allows you to initialzie a MPH on a given list of Ks which are then hashed using the python hashing function, which is modified to only product positive hash values. Similarly to the MyBloomFilter class, the MyMPHF class also allows you to query the MPHF on a given list of Ks' (which are hashed in the function) and a function that calcuates false positves. Calling the funciton testing_mphf will run an example of the MyMPHF running all the functions in the class as well as produce the amount of time it takes to query MPHF, the size of the MPHF, and calculate false positives. 

In order to use this class you much make sure to install these two librarys:
    - pip3 install Cython
    - pip3 install bbhash

Than you can either use the functions independently or run the testing_mphf

Task3 

Task3 greatly builds off Task2 so the two functions created for this task exist in the class MyMPHF and are tested in the function testing_mphf. Running the functions in this class allow you to create a fingerprint array and then query it, given Ks and K's, to gain the false_positve rate

In order to use this class you much make sure to install this library:
    - pip3 install numpy

Than you can either use the functions independently or run the testing_mphf

