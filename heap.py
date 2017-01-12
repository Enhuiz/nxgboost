class MaxHeap(object):
    def __init__(self, key):
        self.l = [0]
        self.key = key

    def push(self, v):
        self.l[0] += 1
        i = self.l[0]

        if len(self.l) == i:
            self.l.append(v)
        else:
            self.l[i] = v

        p = i / 2
        while p >= 1 and self.key(self.l[i]) > self.key(self.l[p]):
            self.l[p], self.l[i] = self.l[i], self.l[p]
            i, p = p, p / 2

    def pop(self):
        if self.l[0] == 0:
            raise Exception("Empty Heap")

        self.l[1], self.l[self.l[0]] = self.l[self.l[0]], self.l[1]
        self.l[0] -= 1

        i = 1
        while True:
            lc = i * 2
            rc = i * 2 + 1

            if lc >= self.l[0]:
                break
            elif rc >= self.l[0]:
                minc = lc
            else:
                minc = lc if self.key(self.l[lc]) > self.key(self.l[rc]) else rc

            if self.key(self.l[minc]) > self.key(self.l[i]):
                self.l[i], self.l[minc] = self.l[minc], self.l[i]
                i = minc
            else:
                break

        return self.l[self.l[0]+1]

    def __len__(self):
        return self.l[0]

    def __str__(self):
        return str([self.key(item) for item in self.l[1:]])