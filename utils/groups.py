import torch
from utils.plotting import *
from sympy.combinatorics.named_groups import SymmetricGroup as sympySG
import math


class Group:
    """
    parent class for all groups
    index: index over groups in family
    order: size of the group
    fourier_order: number of fourier components to use
    """
    def __init__(self, index, order, fourier_order):
        self.index = index
        self.order = order 
        self.fourier_order = fourier_order  
        self.compute_multiplication_table()

    def compose(self, x, y):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError
    
    def compute_multiplication_table(self):
        table = torch.zeros((self.order, self.order), dtype=torch.int64)
        for i in range(self.order):
            for j in range(self.order):
                table[i, j] = self.compose(i, j)
        self.multiplication_table = table
    
    def get_all_data(self, shuffle_seed=False):
        data=torch.zeros((self.order*self.order, 3), dtype=torch.int64)
        for i in range(self.order):
            for j in range(self.order):
                data[i*self.order+j, 0] = i
                data[i*self.order+j, 1] = j
                data[i*self.order+j, 2] = self.multiplication_table[i, j]
        if shuffle_seed:
            torch.manual_seed(shuffle_seed)
            shuffled_indices = torch.randperm(self.order*self.order)
            data = data[shuffled_indices]
        return data
    

    # Fourier basis for cylic-y groups. Should refactor into a different parent class eventually.
    def compute_fourier_basis(self):
        # compute a (frequency, position) tensor encoding the fourier basis
        fourier_basis = []
        fourier_basis.append(torch.ones(self.index)/np.sqrt(self.index))
        fourier_basis_names = ['Const']
        # Note that if p is even, we need to explicitly add a term for cos(kpi), ie 
        # alternating +1 and -1
        for i in range(1, self.fourier_order):
            fourier_basis.append(torch.cos(2*torch.pi*torch.arange(self.index)*i/self.index))
            fourier_basis.append(torch.sin(2*torch.pi*torch.arange(self.index)*i/self.index))
            fourier_basis[-2]/=fourier_basis[-2].norm()
            fourier_basis[-1]/=fourier_basis[-1].norm()
            fourier_basis_names.append(f'cos {i}')
            fourier_basis_names.append(f'sin {i}')

        self.fourier_basis = torch.stack(fourier_basis, dim=0).cuda()
        self.fourier_basis_names = fourier_basis_names  
    
    def animate_fourier_basis(self):
        animate_lines(self.fourier_basis, snapshot_index=self.fourier_basis_names, snapshot='Fourier Component', title='Graphs of Fourier Components (Use Slider)')
    

class CyclicGroup(Group):
    def __init__(self, index):
        super().__init__(index = index, order = index, fourier_order = index//2+1)        
        self.compute_fourier_basis()

    def compose(self, x, y):
        return (x+y)%self.order

    def inverse(self, x):
        return -x%self.order
        

class DihedralGroup(Group):
    """
    Dihedral group of order 2*index. First index elements are rotations, second are reflections.
    i.e. indexed as [e, r, r^2, ..., r^p, s, rs, r^2s, ..., r^ps]
    """
    def __init__(self, index):
        super().__init__(index = index, order = 2*index, fourier_order = index//2+1)        
        self.compute_fourier_basis()

    def idx_to_cpts(self, x):
        r = x % self.index
        # this could be rewritten in a single line
        if x >= self.index: 
            s = 1
        else: 
            s = 0
        return r, s

    def cpts_to_idx(self, r, s):
        return r + s*self.index

    def compose(self, x, y):
        x_r, x_s = self.idx_to_cpts(x)
        y_r, y_s = self.idx_to_cpts(y)
        if x_s == 0:
            z_r = (x_r + y_r) % self.index
            z_s = y_s
        else: # x_s == 1:
            z_r = (x_r - y_r) % self.index
            z_s = (1 + y_s) % 2
        return self.cpts_to_idx(z_r, z_s)

class SymmetricGroup(Group):
    def __init__(self, index):
        self.G = sympySG(index)
        self.order = math.factorial(index)
        super().__init__(index = index, order = self.order, fourier_order = None)
        

    def idx_to_perm(self, x):
        return self.G._elements[x]

    def perm_to_idx(self, perm):
        return self.G._elements.index(perm)

    def compose(self, x, y):
        return self.perm_to_idx(self.idx_to_perm(x) * self.idx_to_perm(y))