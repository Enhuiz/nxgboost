from heap import MaxHeap
import numpy as np

max_heap = MaxHeap(lambda x: x)

for i in xrange(100):
    r = np.random.rand()
    print r
    max_heap.push(r)

print '------------------'

while len(max_heap) > 0:
    print max_heap.pop()