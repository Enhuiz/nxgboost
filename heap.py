class Heap(object):
    def __init__(self, cmp):
        self.l = [0]
        self.cmp = cmp

    def push(self, v):
        self.l[0] += 1
        i = self.l[0]

        if len(self.l) == i:
            self.l.append(v)
        else:
            self.l[i] = v

        p = i / 2
        while p >= 1 and self.cmp(self.l[p], self.l[i]) < 0:
            self.l[p], self.l[i] = self.l[i], self.l[p]
            i, p = p, p / 2

    def pop(self):
        self.l[1], self.l[self.l[0]] = self.l[self.l[0]], self.l[1]
        self.l[0] -= 1

        i = 1
        while True:
            lc = i * 2
            rc = i * 2 + 1

            if lc >= self.l[0]:
                break
            elif rc >= self.l[0]:
                maxc = lc
            else:
                maxc = rc if self.cmp(self.l[lc], self.l[rc]) < 0 else lc

            if self.cmp(self.l[i], self.l[maxc]) < 0:
                self.l[i], self.l[maxc] = self.l[maxc], self.l[i]
                i = maxc
            else:
                break

        return self.l[self.l[0]+1]

    def __len__(self):
        return self.l[0]