import os
import torch
from utils.plotting import *
from sympy.combinatorics.named_groups import SymmetricGroup as sympySG
from sympy.combinatorics import Permutation
import math
from tqdm import tqdm


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
        self.compute_inverses()
        #self.compute_conjugacy_classes()
        #self.compute_element_orders()

    def compose(self, x, y):
        raise NotImplementedError

    def inverse(self, x):
        return (self.multiplication_table[x, :] == self.identity).nonzero().item()
    
    def compute_multiplication_table(self):

        print('Computing multiplication table...')
        filename = f'../utils/cache/S{self.index}_mult_table.pt'

        if os.path.exists(filename):
            print('... loading from file')
            self.multiplication_table = torch.load(filename)
            return

        table = torch.zeros((self.order, self.order), dtype=torch.int64)
        for i in tqdm(range(self.order)):
            for j in range(self.order):
                table[i, j] = self.compose(i, j)
        self.multiplication_table = table

        f = open(filename, 'wb')
        torch.save(table, f)
    
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
    
    def get_subset_of_data(self, indices1, indices2 = 'default', shuffle_seed=False):
        if indices2 == 'default':
            indices2 = indices1
        data=torch.zeros((len(indices1)*len(indices2), 3), dtype=torch.int64)
        for i in range(len(indices1)):
            for j in range(len(indices2)):
                data[i*len(indices1)+j, 0] = indices1[i]
                data[i*len(indices1)+j, 1] = indices2[j]
                data[i*len(indices1)+j, 2] = self.multiplication_table[indices1[i], indices2[j]]
        if shuffle_seed:
            torch.manual_seed(shuffle_seed) 
            shuffled_indices = torch.randperm(len(indices1)*len(indices2))
            data = data[shuffled_indices]
        return data
    
    def compute_conjugacy_classes(self):
        self.conjugacy_classes = []
        seen = set()
        for i in range(self.order):
            if i in seen:
                continue
            current_set = []
            for j in range(self.order):
                # compute jij^-1
                conjugate = self.multiplication_table[self.multiplication_table[j, i], self.inverse(j)].item()
                if conjugate not in current_set:
                    current_set.append(conjugate)
                seen.add(conjugate)
            self.conjugacy_classes.append(current_set)
    
    def compute_element_orders(self):
        orders = []
        for i in range(1):
            print(i)
            current = i
            order = 1
            while current != self.identity:
                current = self.multiplication_table[current, i].item()
                order += 1
            orders.append(order)
        self.orders = orders

    def compute_inverses(self):
        self.inverses = torch.zeros(self.order, dtype=torch.int64)
        for i in range(self.order):
            self.inverses[i] = self.inverse(i)

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
    
    def compute_trivial_rep(self):
        self.trivial_reps = torch.ones(self.order, 1, 1).cuda()
        self.trivial_reps_orth = self.trivial_reps.reshape(self.order, -1)
        self.trivial_reps_orth = torch.linalg.qr(self.trivial_reps_orth)[0]

    def trivial_rep(self, x):
        return self.trivial_reps[x]

class CyclicGroup(Group):
    def __init__(self, index, init_all):
        super().__init__(index = index, order = index, fourier_order = index//2+1)
        self.identity = 0  
        if init_all:
            self.compute_fourier_basis()

    def compose(self, x, y):
        return (x+y)%self.order


class DihedralGroup(Group):
    """
    Dihedral group of order 2*index. First index elements are rotations, second are reflections.
    i.e. indexed as [e, r, r^2, ..., r^p, s, rs, r^2s, ..., r^ps]
    """
    def __init__(self, index):
        super().__init__(index = index, order = 2*index, fourier_order = index//2+1)        
        self.compute_fourier_basis()
        self.identity = 0

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
    def __init__(self, index, init_all):
        self.G = sympySG(index)
        self.order = math.factorial(index)
        self.identity = [i for i in range(self.order) if self.idx_to_perm(i).order() == 1][0]
        super().__init__(index = index, order = self.order, fourier_order = None)
        self.compute_signatures()
        if init_all:
            self.compute_natural_rep()
            self.compute_standard_rep()
            self.compute_standard_sign_rep()
            self.compute_sign_rep()
            self.compute_trivial_rep()
            if self.index == 4:
                self.compute_S4_2d_rep()
            self.all_data = self.get_all_data()[:, :2]
            self.alternating_data = self.get_subset_of_data([i for i in range(self.order) if self.signature(i)==1])[:, :2]
            self.compute_trace_tensor_cubes()

    def idx_to_perm(self, x):
        return self.G._elements[x]

    def perm_to_idx(self, perm):
        return self.G._elements.index(perm)

    def compose(self, x, y):
        return self.perm_to_idx(self.idx_to_perm(x) * self.idx_to_perm(y))

    def perm_order(self, x):
        return self.idx_to_perm(x).order()

    def signature(self, x):
        return self.idx_to_perm(x).signature()
    
    def compute_signatures(self):
        self.signatures = torch.tensor([self.signature(i) for i in range(self.order)]).cuda()
    
    

    # representations

    def compute_natural_rep(self):
        idx = list(np.linspace(0, self.index-1, self.index))
        self.natural_reps = torch.zeros(self.order, self.index, self.index).cuda()
        for x in range(self.order):
            self.natural_reps[x, idx, self.idx_to_perm(x)(idx)] = 1
        self.natural_reps_orth = self.natural_reps.reshape(self.order, self.index*self.index)
        self.natural_reps_orth = torch.linalg.qr(self.natural_reps_orth)[0]


    def compute_standard_rep(self):
        self.standard_reps = []
        basis_transform = torch.zeros(self.index, self.index).cuda()
        for i in range(self.index-1):
            basis_transform[i, i] = 1
            basis_transform[i, i+1] = -1
        basis_transform[self.index-1, self.index-1] = 1 #to make the transform non singular
        for x in self.natural_reps:
            temp = basis_transform @ x @ basis_transform.inverse()
            self.standard_reps.append(temp[:self.index-1, :self.index-1])
        self.standard_reps = torch.stack(self.standard_reps, dim=0).cuda()
        temp = self.standard_reps.reshape(self.order, (self.index-1)*(self.index-1))
        # to orthogonalise, can take the qr decomposition
        # or can take svd and throw away the singular columns
        # these give the same answer
        # s, u, v = torch.linalg.svd(temp)
        # self.standard_sign_reps_orth = u[:, :(self.index)*(self.index)]
        self.standard_reps_orth = torch.linalg.qr(temp)[0]


    def compute_standard_sign_rep(self):
        self.standard_sign_reps = []
        for i in range(self.standard_reps.shape[0]):
            self.standard_sign_reps.append(self.signature(i)*self.standard_reps[i])
        self.standard_sign_reps = torch.stack(self.standard_sign_reps, dim=0).cuda()
        self.standard_sign_reps_orth = self.standard_sign_reps.reshape(self.order, (self.index-1)*(self.index-1))
        self.standard_sign_reps_orth = torch.linalg.qr(self.standard_sign_reps_orth)[0]


    def compute_sign_rep(self):
        self.sign_reps = torch.zeros(self.order, 1, 1).cuda()
        for i in range(self.order):
            self.sign_reps[i, 0, 0] = self.signature(i)
        self.sign_reps_orth = self.sign_reps.reshape(self.order, -1)
        self.sign_reps_orth = torch.linalg.qr(self.sign_reps_orth)[0]

    def compute_S4_2d_rep(self):
        # we just compute the representation here by multiplying out the representation on generators lol 
        # https://arxiv.org/pdf/1112.0687.pdf
        generators = self.G.generators
        rep = {}
        rep[Permutation(0, 1, 2, 3)] = torch.tensor([[-1, -1], [0, 1]]).float() #(0, 1, 2, 3)
        rep[Permutation(3, 2, 1, 0)] = rep[Permutation(0, 1, 2, 3)].inverse()
        rep[Permutation(3)(0,1)] = torch.tensor([[1, 0], [-1, -1]]).float() #(0, 1)

        self.S4_2d_reps = torch.zeros(self.order, 2, 2).cuda()
        for i in range(self.order):
            generator_product = self.G.generator_product(self.idx_to_perm(i), original=True)
            result = torch.eye(2).float()
            for g in generator_product:
                result = result @ rep[g]
            self.S4_2d_reps[i] = result
        self.S4_2d_reps_orth = self.S4_2d_reps.reshape(self.order, 4)
        self.S4_2d_reps_orth = torch.linalg.qr(self.S4_2d_reps_orth)[0]

    def compute_trace_tensor_cube(self, all_data, rep, rep_name):
        print(f'Computing trace tensor cube for {rep_name} representation')
        filename = f'../utils/cache/S{self.index}_{rep_name}_trace_tensor_cube.pt'
        if os.path.exists(filename):
            print('... loading from file')
            return torch.load(filename)
        N = all_data.shape[0]
        t = torch.zeros((self.order*self.order, self.order), dtype=torch.float).cuda()
        for i in tqdm(range(N)):
            x = all_data[i, 0]
            y = all_data[i, 1]
            xy = self.multiplication_table[x, y]
            for z_idx in range(self.order):
                xyz = self.multiplication_table[xy, self.inverses[z_idx]]
                t[i, z_idx] = torch.trace(rep[xyz])
        t = t.reshape(self.order, self.order, self.order)
        f = open(filename, 'wb')
        torch.save(t, f)
        return t

    def compute_trace_tensor_cubes(self):
        # natural rep isnt irreduible
        #self.natural_trace_tensor_cubes = self.compute_trace_tensor_cube(self.all_data, self.natural_rep) 
        #self.natural_trace_tensor_cubes -= self.natural_trace_tensor_cubes.mean(-1)
        self.standard_trace_tensor_cubes = self.compute_trace_tensor_cube(self.all_data, self.standard_reps, 'standard')
        self.standard_trace_tensor_cubes -= self.standard_trace_tensor_cubes.mean(-1, keepdim=True)
        self.standard_sign_trace_tensor_cubes = self.compute_trace_tensor_cube(self.all_data, self.standard_sign_reps, 'standard_sign')
        self.standard_sign_trace_tensor_cubes -= self.standard_sign_trace_tensor_cubes.mean(-1, keepdim=True)
        self.sign_trace_tensor_cubes = self.compute_trace_tensor_cube(self.all_data, self.sign_reps, 'sign')
        self.sign_trace_tensor_cubes -= self.sign_trace_tensor_cubes.mean(-1, keepdim=True)
        self.trivial_trace_tensor_cubes = self.compute_trace_tensor_cube(self.all_data, self.trivial_reps, 'trivial')
        self.trivial_trace_tensor_cubes -= self.trivial_trace_tensor_cubes.mean(-1, keepdim=True)
        if self.index == 4:
            self.S4_2d_trace_tensor_cubes = self.compute_trace_tensor_cube(self.all_data, self.S4_2d_reps, 's4_2d')
            self.S4_2d_trace_tensor_cubes -= self.S4_2d_trace_tensor_cubes.mean(-1, keepdim=True)

    def compute_inverse_reps(self):
        self.standard_inverse_reps = self.compute_inverse_rep(self.standard_reps)
        self.standard_inverse_reps_orth = self.standard_inverse_reps.reshape(self.order, (self.index-1)*(self.index-1))
        self.standard_inverse_reps_orth = torch.linalg.qr(self.standard_inverse_reps_orth)[0]

        self.standard_sign_inverse_reps = self.compute_inverse_rep(self.standard_sign_reps)
        self.standard_sign_inverse_reps_orth = self.standard_sign_inverse_reps.reshape(self.order, (self.index-1)*(self.index-1))
        self.standard_sign_inverse_reps_orth = torch.linalg.qr(self.standard_sign_inverse_reps_orth)[0]

        self.sign_inverse_reps = self.compute_inverse_rep(self.sign_reps)
        self.sign_inverse_reps_orth = self.sign_inverse_reps.reshape(self.order, 1)
        self.sign_inverse_reps_orth = torch.linalg.qr(self.sign_inverse_reps_orth)[0]

        self.trivial_inverse_reps = self.compute_inverse_rep(self.trivial_reps)
        self.trivial_inverse_reps_orth = self.trivial_inverse_reps.reshape(self.order, 1)
        self.trivial_inverse_reps_orth = torch.linalg.qr(self.trivial_inverse_reps_orth)[0]

        if self.index == 4:
            self.S4_2d_inverse_reps = self.compute_inverse_rep(self.S4_2d_reps)
            self.S4_2d_inverse_reps_orth = self.S4_2d_inverse_reps.reshape(self.order, 4)
            self.S4_2d_inverse_reps_orth = torch.linalg.qr(self.S4_2d_inverse_reps_orth)[0]
    

    def compute_inverse_rep(self, rep):
        return rep[self.inverses]



    
