import os
import torch
from utils.plotting import *
from sympy.combinatorics.named_groups import SymmetricGroup as SymPySymmetricGroup
from sympy.combinatorics import Permutation
import math
from tqdm import tqdm


class Group:
    """
    Base class for groups.
    """
    def __init__(self, index, order, fourier_order):
        """
        Initialize a group, compute and store information about it.

        Args:
            index (int): Index of group in wider family.
            order (int): Order of group (number of elements).
            fourier_order (int): Number of fourier representations to compute.
        """
        self.index = index
        self.order = order 
        self.fourier_order = fourier_order  
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
        filename = f'../utils/cache/S{self.index}_mult_table.pt'

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
        inveress = torch.zeros(self.order, dtype=torch.int64)
        for i in range(self.order):
            inverses[i] = self.inverse(i)
        return inverses

    # TODO: refactor this into a "CyclicRepresentation" class

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
        super().__init__(index = index, order = index, fourier_order = index//2+1)
        self.identity = 0  
        if init_all:
            self.compute_fourier_basis()

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
        super().__init__(index = index, order = 2*index, fourier_order = index//2+1)        
        self.compute_fourier_basis()
        self.identity = 0

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

        # initialise parent class
        super().__init__(index = index, order = self.order, fourier_order = None)

        # compute the signatures of the elements
        self.signatures = self.compute_signatures()


        if init_all:

            # get all data to compute metrics
            self.all_data = self.get_all_data()[:, :2]
            self.alternating_data = self.get_subset_of_data([i for i in range(self.order) if self.signature(i)==1])[:, :2]

            # parameters for representation initialisation
            rep_params = {
                'group_index': self.index,
                'group_order': self.order,
                'multiplication_table': self.multiplication_table,
                'inverses': self.inverses,
                'all_data': self.all_data,
            }

            # initialise representations
            sign_rep = SignRepresentation([], **rep_params)
            natural_rep = NaturalRepresentation([], **rep_params)
            standard_rep = StandardRepresentation([natural_rep.rep], **rep_params)
            standard_sign_rep = StandardSignRepresentation([standard_rep.rep, sign_rep.rep], **rep_params)

            self.irreps = {
                'sign': sign_rep,
                'standard': standard_rep,
                'standard_sign': standard_sign_rep,
            }

            if self.index == 4:

                s4_2d_generators = {}
                s4_2d_generators[Permutation(0, 1, 2, 3)] = torch.tensor([[-1, -1], [0, 1]]).float() 
                s4_2d_generators[Permutation(3, 2, 1, 0)] = s4_2d_generators[Permutation(0, 1, 2, 3)].inverse()
                s4_2d_generators[Permutation(3)(0,1)] = torch.tensor([[1, 0], [-1, -1]]).float() 
                s4_2d_rep = S4_2d_Representation([s4_2d_generators, self.G], **rep_params)
                self.irreps['s4_2d'] = s4_2d_rep
            
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
                s6_5d_a_rep = S6_5d_a_Representation([s6_5d_a_generators, self.G], **rep_params)
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
                s6_5d_b_rep = S6_5d_b_Representation([s6_5d_b_generators, self.G], **rep_params)
                self.irreps['s6_5d_b'] = s6_5d_b_rep




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


class SymmetricRepresentation():
    """
    Base class for all representations of a symmetric group.
    """

    def __init__(self, compute_rep_params, index, order, multiplication_table, inverses):
        """
        Initialise the symmetric group representation.

        Args:
            compute_rep_params (tuple): representation specific parameters required for computing the representation
            index (int): group index in family of symmetric groups
            order (int): order of the group
            multiplication_table (torch.tensor): square (group.order, group.order) tensor of group multiplication table 
            inverses (torch.tensor): vector of group inverses
        """

        self.friendly_name = 'none'
        self.dim = None
        self.index = index
        self.order = order
        self.multiplication_table = multiplication_table
        self.inverses = inverses

        self.rep = self.compute_rep(*compute_rep_params)
        self.orth_rep = self.compute_orth_rep(self.rep)

        self.logit_trace_tensor_cube = self.compute_logit_trace_tensor_cube()

        self.inverse_rep = self.compute_inverse_rep()
        self.inverse_orth_rep = self.compute_orth_rep(self.inverse_rep)

    def compute_rep():
        """
        Compute the representation. Must be implemented by child class.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
    
    def compute_orth_rep(rep):
        """
        Use QR decomposition to orthogonalise the representation but retain the subspace spanned by the columns. 

        Args:
            rep (torch.tensor): (group.order, dim^2) tensor of representation

        Returns:
            torch.tensor: (group.order, dim^2) tensor with orthonormal columns en
        """
        orth_rep = rep.reshape(self.order, self.dim * self.dim)
        orth_rep = torch.linalg.qr(orth_rep)[0]
        return orth_rep


    def compute_logit_trace_tensor_cube(self):
        """
        Under the hypothesis, the network computes tr(\rho(x)\rho(y)\rho(z^-1)) for some representation \rho.
        This function computes this trace tensor cube for a given representation, and returns this tensor, centred around 0 in the 
        logit space.

        Returns:
            torch.tensor: (group.order^3) trace tensor cube
        """
        print(f'Computing trace tensor cube for {self.friendly_name} representation')
        filename = f'../utils/cache/S{self.index}_{self.friendly_name}_trace_tensor_cube.pt'
        if os.path.exists(filename):
            print('... loading from file')
            return torch.load(filename)
        N = all_data.shape[0]
        t = torch.zeros((self.order*self.order, self.order), dtype=torch.float).cuda()
        for i in tqdm(range(N)):
            x = self.all_data[i, 0]
            y = self.all_data[i, 1]
            xy = self.multiplication_table[x, y]
            for z_idx in range(self.order):
                xyz = self.multiplication_table[xy, self.inverses[z_idx]]
                t[i, z_idx] = torch.trace(self.rep[xyz])
        t = t.reshape(self.order, self.order, self.order)
        f = open(filename, 'wb')
        torch.save(t, f)
        return t - t.mean(-1, keepdim=True)
    
    def compute_inverse_rep(self):
        """
        Compute the inverse representation.

        Returns:
            torch.tensor: (group.order, dim^2) tensor of inverse representations
        """
        return self.rep[self.inverses]
    



class NaturalRepresentation(SymmetricRepresentation):
    def __init__(self, compute_rep_params, init_rep_params):
        super().__init__(compute_rep_params, **init_rep_params)
        self.friendly_name = 'natural'
        self.dim = index
    
    def compute_rep(self):
        idx = list(np.linspace(0, self.group_index-1, self.group_index))
        rep = torch.zeros(self.order, self.group_index, self.group_index).cuda()
        for x in range(self.order):
            rep[x, idx, self.idx_to_perm(x)(idx)] = 1
        return rep

class StandardRepresentation(SymmetricRepresentation):
    def __init__(self, compute_rep_params, init_rep_params):
        super().__init__(compute_rep_params, **init_rep_params)
        self.friendly_name = 'standard'
        self.dim = self.index - 1
    
    def compute_rep(self, natural_reps):
        rep = []
        basis_transform = torch.zeros(self.index, self.index).cuda()
        for i in range(self.index-1):
            basis_transform[i, i] = 1
            basis_transform[i, i+1] = -1
        basis_transform[self.index-1, self.index-1] = 1 #to make the transform non singular
        for x in natural_reps:
            temp = basis_transform @ x @ basis_transform.inverse()
            rep.append(temp[:self.index-1, :self.index-1])
        rep = torch.stack(rep, dim=0).cuda()
        return rep        
        
class SignRepresentation(SymmetricRepresentation):
    def __init__(self, compute_rep_params, init_rep_params):
        super().__init__(compute_rep_params, **init_rep_params)
        self.friendly_name = 'sign'
        self.dim = 1

    def compute_rep(self, signatures):
        rep = torch.zeros(self.order, 1, 1).cuda()
        rep[:, 0, 0] = signatures
        return rep

class StandardSignRepresentation(SymmetricRepresentation):
    def __init__(self, compute_rep_params, init_rep_params):
        super().__init__(compute_rep_params, **init_rep_params)
        self.friendly_name = 'standard_sign'
        self.dim = self.index - 1
        

    def compute_rep(self, standard_reps, signatures):
        rep = []
        for i in range(standard_reps.shape[0]):
            rep.append(signatures[i]*standard_reps[i])
        rep = torch.stack(rep, dim=0).cuda()
        return rep

class SymmetricRepresentationFromGenerators():
    def __init__(self, compute_rep_params, init_rep_params, name):
        super().__init__(compute_rep_params, **init_rep_params)
        self.friendly_name = 'from_generators'

        # TODO: make this less hacky
        self.dim = compute_rep_params[0].values[0].shape[0] # hacky way to get the dimension of the representation

    def compute_rep(self, generators, G):
        rep = torch.zeros(self.order, self.dim, self.dim).cuda()
        for i in range(self.order):
            generator_product = self.G.generator_product(self.idx_to_perm(i), original=True)
            result = torch.eye(self.dim).float().cuda()
            for g in generator_product:
                result = result @ self.generators[g]
            reps[i] = result
        return reps