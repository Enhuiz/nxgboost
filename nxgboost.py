import numpy as np
from heap import MaxHeap

class SquareLoss(object):
    def __init__(self):
        pass

    def G(self, prediction, target):
        return 2 * (prediction - target)
        
    def H(self, prediction, target):
        return 2        

class NXGBoost(object):
    def __init__(self):
        self.loss_func = SquareLoss()
        self.roots = []

    def fit(self, x, y, n_estimators=100, eta=0.1, lambd=0.1, max_depth=5):
        samples = self.init_samples(x, y)
        for _ in xrange(n_estimators):
            root = self.build_tree(eta, lambd, max_depth, samples)
            self.roots.append(root)
            self.refresh_prediction(root)
    
    def predict(self, x):
        return np.array([np.sum([self.weight(root, row) for root in self.roots]) for row in x])

    def init_samples(self, x, y): # x: n*m, y: n*1
        samples = []
        for i in xrange(len(x)):
            samples.append({
                'features': x[i],
                'target': float(y[i]),
                'prediction': 0,
            })
        return samples

    def build_tree(self, eta, lambd, max_depth, samples):
        root = {
            'k': None,
            'w': None,
            'val': None,
            'lc': None,
            'rc': None,
            'depth': 1,
            'samples': samples,
        }

        G = np.sum([self.loss_func.G(sample['prediction'], sample['target']) 
            for sample in samples])
        H = np.sum([self.loss_func.H(sample['prediction'], sample['target'])
            for sample in samples])

        max_heap = MaxHeap(key=lambda x: x[0])
        max_heap.push(self.split_attempt(root, eta, lambd, G, H))

        while len(max_heap) > 0:
            gain, node, lc, rc, k, val = max_heap.pop()

            if gain == -np.inf:
                break

            if node['depth'] >= max_depth:
                break

            node['lc'] = lc
            node['rc'] = rc
            node['w'] = None
            node['k'] = k
            node['val'] = val

            max_heap.push(self.split_attempt(lc, eta, lambd, G, H))
            max_heap.push(self.split_attempt(rc, eta, lambd, G, H))

        return root

    def split_attempt(self, node, eta, lambd, G, H):
        samples = node['samples']

        max_gain = -np.inf
        split_k = None
        split_val = None
        split_samples_l = None
        split_samples_r = None
        split_w_l = None
        split_w_r = None

        for k in xrange(len(samples[0]['features'])):
            GL, HL = 0, 0
            sorted_samples = sorted(samples, key=lambda sample: sample['features'][k])
            for j in xrange(len(sorted_samples)):
                sample = sorted_samples[j]
                GL += self.loss_func.G(sample['prediction'], sample['target'])
                HL += self.loss_func.H(sample['prediction'], sample['target'])
                GR = G - GL
                HR = H - HL
                gain = GL**2/(HL + lambd) + GR**2/(HR + lambd) - G**2/(H + lambd)
                if max_gain < gain:
                    split_k = k
                    split_val = sample['features'][k]
                    split_w_l = -GL/(HL + lambd) * eta
                    split_w_r = -GR/(HR + lambd) * eta
                    split_samples_l = sorted_samples[:j]
                    split_samples_r = sorted_samples[j:]
                    max_gain = gain

        children = []

        if len(split_samples_l) != 0:
            children.append({
                'w': split_w_l,
                'k': None,
                'val': None,
                'lc': None,
                'rc': None,
                'depth': node['depth'] + 1,
                'samples': split_samples_l,
            })

        if len(split_samples_r) != 0:
            children.append({
                'w': split_w_r,
                'k': None,
                'val': None,
                'lc': None,
                'rc': None,
                'depth': node['depth'] + 1,
                'samples': split_samples_r,
            })

        # The gain is negatived to fit the min heap
        if len(children) == 2:
            return max_gain, node, children[0], children[1], split_k, split_val
        else:
            return -np.inf, None, None, None, None, None

    def refresh_prediction(self, node):
        if node is None:
            return
        elif node['w'] is not None:
            for sample in node['samples']:
                sample['prediction'] += node['w']
        else:
            self.refresh_prediction(node['lc'])
            self.refresh_prediction(node['rc'])

    def weight(self, node, x):
        if node['k'] is None:
            return node['w'] or 0
        else:
            if x[node['k']] < node['val']:
                return self.weight(node['lc'], x)
            else:
                return self.weight(node['rc'], x)