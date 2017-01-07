import numpy as np

class SquareLoss(object):
    def __init__(self):
        pass

    def G(self, target, prediction):
        return 2 * (prediction - target)
        
    def H(self, target, prediction):
        return 2
                        
class NXGBoost(object):
    def __init__(self):
        self.loss_func = SquareLoss()
        self.roots = []

    def fit(self, x, y, n_estimators=100, lambd=0.1, max_depth=2):
        samples = self.init_samples(x, y)
        for _ in xrange(n_estimators):
            root = self.build_tree(max_depth, lambd, samples)
            self.roots.append(root)
            self.refresh_prediction(root)
    
    def predict(self, x):
        return np.array([np.sum([self.score(root, row) for root in self.roots]) for row in x])

    def init_samples(self, x, y): # x: n*m, y: n*1
        samples = []
        for i in xrange(len(x)):
            samples.append({
                'features': x[i],
                'target': y[i],
                'prediction': 0,
            })
        return samples

    def build_tree(self, depth, lambd, samples):
        root = {
            'k': None,
            'w': None,
            'val': None,
            'lc': None,
            'rc': None,
            'samples': samples,
        }
        G = np.sum([self.loss_func.G(sample['target'], sample['prediction']) 
            for sample in samples])
        H = np.sum([self.loss_func.H(sample['target'], sample['prediction'])
            for sample in samples])
        self.build_tree_helper(root, depth, lambd, G, H)
        return root

    def build_tree_helper(self, node, depth, lambd, G, H):
        if node is None or node['samples'] is None or depth <= 0:
            return
        self.split(node, lambd, G, H)
        self.build_tree_helper(node['lc'], depth-1, lambd, G, H)
        self.build_tree_helper(node['rc'], depth-1, lambd, G, H)

    def refresh_prediction(self, node):
        if node is None:
            return
        elif node['w'] is not None:
            for sample in node['samples']:
                sample['prediction'] += node['w']
        else:
            self.refresh_prediction(node['lc'])
            self.refresh_prediction(node['rc'])


    def split(self, node, lambd, G, H):
        samples = node['samples']

        max_score = -np.inf
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
                GL += self.loss_func.G(sample['target'], sample['prediction'])
                HL += self.loss_func.H(sample['target'], sample['prediction'])
                GR = G - GL
                HR = G - HL
                score = GL**2/(HL + lambd) + GR**2/(GR + lambd) - G**2/(H + lambd)
                if max_score < score:
                    split_k = k
                    split_val = sample['features'][k]
                    split_w_l = - GL / (HL + lambd)
                    split_w_r = - GR / (HR + lambd)
                    split_samples_l = sorted_samples[:j]
                    split_samples_r = sorted_samples[j:]
                    max_score = score
        children = []
        if len(split_samples_l) != 0:
            children.append({
                'w': split_w_l,
                'k': None,
                'val': None,
                'lc': None,
                'rc': None,
                'samples': split_samples_l,
            })
        if len(split_samples_r) != 0:
            children.append({
                'w': split_w_r,
                'k': None,
                'val': None,
                'lc': None,
                'rc': None,
                'samples': split_samples_r,
            })
        if len(children) == 2:
            node['w'] = None
            node['k'] = split_k
            node['val'] = split_val
            node['lc'] = children[0]
            node['rc'] = children[1]

    def score(self, node, x):
        if node['k'] is None:
            return node['w'] or 0
        else:
            if x[node['k']] < node['val']:
                return self.score(node['lc'], x)
            else:
                return self.score(node['rc'], x)