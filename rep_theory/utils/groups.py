import os
import torch
from utils.plotting import *
from sympy.combinatorics.named_groups import SymmetricGroup as SymPySymmetricGroup
from sympy.combinatorics import Permutation
import math
from tqdm import tqdm
from utils.representations import *
import math

class Group:
    """
    Base class for groups.
    """
    def __init__(self, index, order):
        """
        Initialize a group, compute and store information about it.

        Args:
            index (int): Index of group in wider family.
            order (int): Order of group (number of elements).
            fourier_order (int): Number of fourier representations to compute.
        """
        self.index = index
        self.order = order 
        self.multiplication_table = self.compute_multiplication_table()
        self.inverses = self.compute_inverses()
        #self.compute_conjugacy_classes()
        #self.compute_element_orders()

    def compose(self, x, y):
        """
        Compose two elements of the group.

        Args:
            x (int): Index of left element
            y (int): Index of right element

        Raises:
            NotImplementedError: Must be implemented by child class.
        """
        raise NotImplementedError

    def inverse(self, x):
        """
        Compute the inverse of an element of the group.

        Args:
            x (int): Index of element to inverse

        Returns:
            int: Index of inverse element
        """
        return (self.multiplication_table[x, :] == self.identity).nonzero().item()
    
    def compute_multiplication_table(self):
        """
        Compute the multiplication table of the group. Caches/loads from file if possible.
        """
        print('Computing multiplication table...')
        filename = f'utils/cache/{self.acronym}{self.index}/{self.acronym}{self.index}_mult_table.pt'

        if os.path.exists(filename):
            print('... loading from file')
            table = torch.load(filename)
        else:
            table = torch.zeros((self.order, self.order), dtype=torch.int64).cuda()
            for i in tqdm(range(self.order)):
                for j in range(self.order):
                    table[i, j] = self.compose(i, j)
            f = open(filename, 'wb')
            torch.save(table, f)
        return table

        
    
    def get_all_data(self, shuffle_seed=False):
        """
        Get's all data and labels for the pairwise composition task.

        Args:
            shuffle_seed (bool, optional): Shuffle data for training. Defaults to False.

        Returns:
            torch.tensor: Tensor of shape (order*order, 3) where each row is (x, y, x*y).
        """
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
        """
        Gets a subset of data and labels for the pairwise composition task.
        TODO: make this more efficient and combine with the above function

        Args:
            indices1 (_type_): _description_
            indices2 (str, optional): _description_. Defaults to 'default'.
            shuffle_seed (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
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
        """
        Computes the conjugacy classes of the group.
        """
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
        """
        Compute the order of each element of the group.
        """
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
        inverses = torch.zeros(self.order, dtype=torch.int64)
        for i in range(self.order):
            inverses[i] = self.inverse(i)
        return inverses

    # TODO: refactor this into a "CyclicRepresentation" class

    # Fourier basis for cylic-y groups. Should refactor into a different parent class eventually.
    # def compute_fourier_basis(self):
    #     # compute a (frequency, position) tensor encoding the fourier basis
    #     fourier_basis = []
    #     fourier_basis.append(torch.ones(self.index)/np.sqrt(self.index))
    #     fourier_basis_names = ['Const']
    #     # Note that if p is even, we need to explicitly add a term for cos(kpi), ie 
    #     # alternating +1 and -1
    #     for i in range(1, self.fourier_order):
    #         fourier_basis.append(torch.cos(2*torch.pi*torch.arange(self.index)*i/self.index))
    #         fourier_basis.append(torch.sin(2*torch.pi*torch.arange(self.index)*i/self.index))
    #         fourier_basis[-2]/=fourier_basis[-2].norm()
    #         fourier_basis[-1]/=fourier_basis[-1].norm()
    #         fourier_basis_names.append(f'cos {i}')
    #         fourier_basis_names.append(f'sin {i}')

    #     self.fourier_basis = torch.stack(fourier_basis, dim=0).cuda()
    #     self.fourier_basis_names = fourier_basis_names  
    
    # def animate_fourier_basis(self):
    #     animate_lines(self.fourier_basis, snapshot_index=self.fourier_basis_names, snapshot='Fourier Component', title='Graphs of Fourier Components (Use Slider)')
    
    # def compute_trivial_rep(self):
    #     self.trivial_reps = torch.ones(self.order, 1, 1).cuda()
    #     self.trivial_reps_orth = self.trivial_reps.reshape(self.order, -1)
    #     self.trivial_reps_orth = torch.linalg.qr(self.trivial_reps_orth)[0]

    # def trivial_rep(self, x):
    #     return self.trivial_reps[x]

class CyclicGroup(Group):
    """
    Class for cyclic groups, i.e. the modular addition task.
    """
    def __init__(self, index, init_all=False):
        """
        Initiliazes a cyclic group of order index.

        Args:
            index (int): Index and order of the group.
            init_all (bool, optional): If false, only calculate what is required to train. If true, calculate all other tensors needed to track metrics. Defaults to False.
        """
        self.identity = 0  
        self.acronym = 'C'
        super().__init__(index = index, order = index)

        if init_all:

            # get all data to compute metrics
            self.all_data = self.get_all_data()[:, :2]

            # parameters for representation initialisation
            rep_params = {
                'index': self.index,
                'order': self.order,
                'multiplication_table': self.multiplication_table,
                'inverses': self.inverses,
                'all_data': self.all_data,
                'group_acronym': self.acronym
            }

            # initialise representations
            self.irreps = {}
            
            # trivial representation
            trivial_rep = TrivialRepresentation([], rep_params)
            self.irreps['trivial'] = trivial_rep

            # if order is even, add the sign representation
            if self.order%2 == 0:
                # compute the signatures, alternating +1 and -1 
                signatures = torch.ones(self.order)
                signatures[1::2] = -1
                sign_rep = SignRepresentation([signatures], rep_params)
                self.irreps['sign'] = sign_rep
        
            # 2d representations
            for k in range(1, math.ceil(self.order/2)):
                name = f'freq_{k}'
                rep = Cyclic2dRepresentation([k], rep_params, name)
                self.irreps[name] = rep

            

            # copy over the non-trivial irreps    
            self.non_trivial_irreps = self.irreps.copy()
            del self.non_trivial_irreps['trivial']

    def compose(self, x, y):
        """
        Compose two elements of the group.

        Args:
            x (int): Left index 
            y (int): Right index

        Returns:
            int: index of composition of x and y
        """
        return (x+y)%self.order


class DihedralGroup(Group):
    """
    Dihedral group of order 2*index. First half of the elements are rotations, second are reflections. 
    i.e. indexed as [e, r, r^2, ..., r^p, s, rs, r^2s, ..., r^ps]
    """
    def __init__(self, index):
        """
        Initialise the group.

        Args:
            index (int): Index of the group. Order is 2*index.
        """
        super().__init__(index = index, order = 2*index)        
        self.compute_fourier_basis()
        self.identity = 0
        self.acronym = 'D'

    def idx_to_cpts(self, x):
        """
        Convert an index to a tuple of (rotation, reflection) components.

        Args:
            x (int): index of element in group

        Returns:
            (int, int): tuple (r, s) where r \in [0, index-1] and s \in {0, 1}
        """
        r = x % self.index
        if x >= self.index: 
            s = 1
        else: 
            s = 0
        return r, s

    def cpts_to_idx(self, r, s):
        """
        Convert a tupl of (rotation, reflection) components to an index.

        Args:
            r (int): rotation component \in [0, index-1]
            s (int): reflection component \in {0, 1}

        Returns:
            int: index of element
        """
        return r + s*self.index

    def compose(self, x, y):
        """
        Compose elements of the group by converting to components, composing, and converting back.

        Args:
            x (int): left index
            y (int): right index

        Returns:
            int: composition index
        """
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
    """
    Class for the symmetric group of order index.
    """
    def __init__(self, index, init_all=False):
        """
        Initialise the group. Optionally calculate all other tensors needed to track metrics.

        Args:
            index (int): Index of group in family of symmetric groups.
            init_all (bool, optional): If false, only calculate what is required to train. If true, calculate all other tensors needed to track metrics. Defaults to False.
        """

        self.order = math.factorial(index)

        # use sympy to generate the group
        self.G = SymPySymmetricGroup(index)

        # hacky method to find the index of the identity element 
        self.identity = [i for i in range(self.order) if self.idx_to_perm(i).order() == 1][0]

        self.acronym = 'S'

        # initialise parent class
        super().__init__(index = index, order = self.order)

        # compute the signatures of the elements
        self.signatures = self.compute_signatures()


        if init_all:

            # get all data to compute metrics
            self.all_data = self.get_all_data()[:, :2]
            self.alternating_data = self.get_subset_of_data([i for i in range(self.order) if self.signature(i)==1])[:, :2]

            # parameters for representation initialisation
            rep_params = {
                'index': self.index,
                'order': self.order,
                'multiplication_table': self.multiplication_table,
                'inverses': self.inverses,
                'all_data': self.all_data,
                'group_acronym': self.acronym
            }

            # initialise representations
            trivial_rep = TrivialRepresentation([], rep_params)
            sign_rep = SignRepresentation([self.signatures], rep_params)
            natural_rep = NaturalRepresentation([self.idx_to_perm], rep_params)
            standard_rep = StandardRepresentation([natural_rep.rep], rep_params)
            standard_sign_rep = StandardSignRepresentation([standard_rep.rep, sign_rep.rep], rep_params)

            self.irreps = {
                'trivial': trivial_rep,
                'sign': sign_rep,
                'standard': standard_rep,
                'standard_sign': standard_sign_rep,
            }


            self.other_reps = {
                'natural': natural_rep,
            }

            if self.index == 4:

                s4_2d_generators = {}
                s4_2d_generators[Permutation(0, 1, 2, 3)] = torch.tensor([[-1, -1], [0, 1]]).float() 
                s4_2d_generators[Permutation(3, 2, 1, 0)] = s4_2d_generators[Permutation(0, 1, 2, 3)].inverse()
                s4_2d_generators[Permutation(3)(0,1)] = torch.tensor([[1, 0], [-1, -1]]).float() 
                s4_2d_rep = SymmetricRepresentationFromGenerators([s4_2d_generators, self.G, self.idx_to_perm], rep_params, 's4_2d')
                self.irreps['s4_2d'] = s4_2d_rep

            if self.index == 5:
                # 2,2,1 specht
                s5_5d_a_generators = {}
                s5_5d_a_generators[Permutation(0, 1, 2, 3, 4)] = torch.tensor([
                    [ 1, -1, -1,  1,  0],
                    [ 0, -1, -1,  0,  1],
                    [ 1, -1,  0,  0,  0],
                    [ 0, -1,  0,  0,  0],
                    [ 1, -1, -1,  0,  0]
                ]).float()
                s5_5d_a_generators[Permutation(4, 3, 2, 1, 0)] = s5_5d_a_generators[Permutation(0, 1, 2, 3, 4)].inverse()
                s5_5d_a_generators[Permutation(4)(0,1)] = torch.tensor([
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  0, -1,  0],
                    [-1,  0,  0,  0,  0],
                    [ 0, -1,  0,  0,  0],
                    [ 0,  0,  0,  0, -1]
                ]).float()
                s5_5d_a_rep = SymmetricRepresentationFromGenerators([s5_5d_a_generators, self.G, self.idx_to_perm], rep_params, 's5_5d_a')
                self.irreps['s5_5d_a'] = s5_5d_a_rep

                # 3,2 specht
                s5_5d_b_generators = {}
                s5_5d_b_generators[Permutation(0, 1, 2, 3, 4)] = torch.tensor([
                    [-1,  1, -1,  0,  0],
                    [ 0,  0,  0,  1, -1],
                    [ 0,  0,  1,  0, -1],
                    [ 1,  0,  0,  0,  0],
                    [ 1,  0,  1,  0,  0]
                ]).float()
                s5_5d_b_generators[Permutation(4, 3, 2, 1, 0)] = s5_5d_b_generators[Permutation(0, 1, 2, 3, 4)].inverse()
                s5_5d_b_generators[Permutation(4)(0,1)] = torch.tensor([
                    [ 1,  0,  0,  0,  0],
                    [ 0,  0,  0, -1,  0],
                    [-1,  0,  0,  0, -1],
                    [ 0, -1,  0,  0,  0],
                    [-1,  0, -1,  0,  0]
                ]).float()
                s5_5d_b_rep = SymmetricRepresentationFromGenerators([s5_5d_b_generators, self.G, self.idx_to_perm], rep_params, 's5_5d_b')
                self.irreps['s5_5d_b'] = s5_5d_b_rep

                # 3,1,1 specht
                s5_6d_generators = {}
                s5_6d_generators[Permutation(0, 1, 2, 3, 4)] = torch.tensor([
                    [ 1, -1,  1,  0,  0,  0],
                    [ 1,  0,  0, -1,  1,  0],
                    [ 0,  1,  0, -1,  0,  1],
                    [ 1,  0,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0,  0],
                    [ 0,  0,  0,  1,  0,  0]
                ]).float()
                s5_6d_generators[Permutation(4, 3, 2, 1, 0)] = s5_6d_generators[Permutation(0, 1, 2, 3, 4)].inverse()
                s5_6d_generators[Permutation(4)(0,1)] = torch.tensor([
                    [-1,  0,  0,  0,  0,  0],
                    [ 0,  0,  0, -1,  0,  0],
                    [ 0,  0,  0,  0, -1,  0],
                    [ 0, -1,  0,  0,  0,  0],
                    [ 0,  0, -1,  0,  0,  0],
                    [ 0,  0,  0,  0,  0,  1]
                ]).float()
                s5_6d_rep = SymmetricRepresentationFromGenerators([s5_6d_generators, self.G, self.idx_to_perm], rep_params, 's5_6d')
                self.irreps['s5_6d'] = s5_6d_rep



            
            if self.index == 6:
                
                # (3,3) specht rep
                s6_5d_a_generators = {}
                s6_5d_a_generators[Permutation(0, 1, 2, 3, 4, 5)] = torch.tensor([
                    [-1,  1, -1,  0,  0],
                    [ 0,  0,  0,  1, -1],
                    [ 0,  0,  1,  0, -1],
                    [ 0, -1,  1,  1, -1],
                    [ 0, -1,  1,  0, -1]]).float() 
                s6_5d_a_generators[Permutation(5, 4, 3, 2, 1, 0)] = s6_5d_a_generators[Permutation(0, 1, 2, 3, 4, 5)].inverse()
                s6_5d_a_generators[Permutation(5)(0,1)] = torch.tensor([
                    [ 1,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [-1,  1,  0, -1,  0],
                    [-1,  0,  1,  0, -1]],
                ).float() 
                s6_5d_a_rep = SymmetricRepresentationFromGenerators([s6_5d_a_generators, self.G, self.idx_to_perm], rep_params, 's6_5d_a')
                self.irreps['s6_5d_a'] = s6_5d_a_rep

                # (2,2,2) specht rep
                s6_5d_b_generators = {}
                s6_5d_b_generators[Permutation(0, 1, 2, 3, 4, 5)] = torch.tensor([
                    [ 1, -1, -1,  1,  0],
                    [ 0, -1, -1,  0,  1],
                    [ 0,  0, -1,  1,  0],
                    [ 0,  0, -1,  0,  1],
                    [ 0, -1, -1,  1,  1]]).float() 
                s6_5d_b_generators[Permutation(5, 4, 3, 2, 1, 0)] = s6_5d_b_generators[Permutation(0, 1, 2, 3, 4, 5)].inverse()
                s6_5d_b_generators[Permutation(5)(0,1)] = torch.tensor([
                    [ 1,  0,  0,  0,  0],
                    [ 0,  1,  0,  0,  0],
                    [ 1,  0, -1,  0,  0],
                    [ 0,  1,  0, -1,  0],
                    [ 1,  1,  0,  0, -1]]
                ).float() 
                s6_5d_b_rep = SymmetricRepresentationFromGenerators([s6_5d_b_generators, self.G, self.idx_to_perm], rep_params, 's6_5d_b')
                self.irreps['s6_5d_b'] = s6_5d_b_rep

            # copy over the non-trivial irreps    
            self.non_trivial_irreps = self.irreps.copy()
            del self.non_trivial_irreps['trivial']
    

    def idx_to_perm(self, x):
        """
        Convert an index to a permutation.

        Args:
            x (int): index of element in group

        Returns:
            Permutation: permutation object from sympy
        """
        return self.G._elements[x]

    def perm_to_idx(self, perm):
        """
        Converts a permutation to an index.

        Args:
            perm (Permutation): permutation object from sympy

        Returns:
            int: index of element in group
        """
        return self.G._elements.index(perm)

    def compose(self, x, y):
        """
        Compose elements of the group by converting to permutations, composing, and converting back.

        Args:
            x (int): left index
            y (int): right index

        Returns:
            int: index of composition
        """
        return self.perm_to_idx(self.idx_to_perm(x) * self.idx_to_perm(y))

    def perm_order(self, x):
        """
        Gets the order of a permutation.

        Args:
            x (int): index of element

        Returns:
            int: order of permutation
        """
        return self.idx_to_perm(x).order()

    def signature(self, x):
        """
        Gets the signature of a permutation.

        Args:
            x (int): index of element

        Returns:
            int: Integer \in {0, 1} representing the signature of the permutation.
        """
        return self.idx_to_perm(x).signature()
    
    def compute_signatures(self):
        """
        Compute and store the signature of each element in the group.

        Returns:
            torch.tensor: tensor of signatures
        """
        signatures = torch.tensor([self.signature(i) for i in range(self.order)]).cuda()
        return signatures


