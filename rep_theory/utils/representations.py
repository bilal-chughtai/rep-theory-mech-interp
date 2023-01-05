import torch
import os
from tqdm import tqdm
import numpy as np

class Representation():
    """
    Base class for all representations.
    """

    def __init__(self, compute_rep_params, index, order, multiplication_table, inverses, all_data, group_acronym, irrep=True):
        """
        Initialise the symmetric group representation.

        Args:
            compute_rep_params (tuple): representation specific parameters required for computing the representation
            index (int): group index in family
            order (int): order of the group
            multiplication_table (torch.tensor): square (group.order, group.order) tensor of group multiplication table 
            inverses (torch.tensor): vector of group inverses
            all_data ()
            irrep (Boolean)
        """

        self.index = index
        self.order = order
        self.multiplication_table = multiplication_table
        self.inverses = inverses
        self.all_data = all_data
        self.group_acronym = group_acronym

        # TODO: this is needed to get the dimension of generated representations - think up a better way of doing this
        self.compute_rep_params = compute_rep_params

        self.dim = self.get_rep_dim()

        self.rep = self.compute_rep(*compute_rep_params)

        if irrep:
            self.orth_rep = self.compute_orth_rep(self.rep)
            self.logit_trace_tensor_cube = self.compute_logit_trace_tensor_cube()


    def get_rep_dim(self):
        return NotImplementedError

    def compute_rep():
        """
        Compute the representation. Must be implemented by child class.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
    
    def compute_orth_rep(self, rep):
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
        This function computes this trace tensor cube for a given representation, and returns this tensor
        
        WE DONT CENTRE.

        Returns:
            torch.tensor: (group.order^3) trace tensor cube
        """
        print(f'Computing trace tensor cube for {self.friendly_name} representation')
        filename = f'utils/cache/{self.group_acronym}{self.index}/{self.group_acronym}{self.index}_{self.friendly_name}_trace_tensor_cube.pt'
        if os.path.exists(filename):
            print('... loading from file')
            t = torch.load(filename)
            return t #- t.mean(-1, keepdim=True)
        N = self.all_data.shape[0]
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
        return t #- t.mean(-1, keepdim=True)


class TrivialRepresentation(Representation):
    """
    The trivial representation of the symmetric group.
    """
    def __init__(self, compute_rep_params, init_rep_params):
        """
        Initialise the trivial representation. 

        Args:
            compute_rep_params (list): idx_to_perm function required to compute the representation
            init_rep_params (dict): standard group parameters needed by the representation, including index, order, multiplication_table, inverses, all_data
        """
        self.friendly_name = 'trivial'
        super().__init__(compute_rep_params, **init_rep_params, irrep=True)
    
    def get_rep_dim(self):
        """
        Get the dimension of the representation.

        Returns:
            int: dimension of the representation
        """
        return 1

    def compute_rep(self):
        """
        Compute the trivial representation.

        Args:
            idx_to_perm (function): function to convert an index to a permutation

        Returns:
            torch.tensor: (group.order, 1) tensor of representation
        """
        return torch.ones((self.order, 1, 1), dtype=torch.float).cuda()


class NaturalRepresentation(Representation):
    """
    Compute the natural representation of the symmetric group.
    """
    def __init__(self, compute_rep_params, init_rep_params):
        """
        Initialise the natural representation. 

        Args:
            compute_rep_params (list): idx_to_perm function required to compute the representation
            init_rep_params (dict): standard group parameters needed by the representation, including index, order, multiplication_table, inverses, all_data
        """
        self.friendly_name = 'natural'
        super().__init__(compute_rep_params, **init_rep_params, irrep=False)
    
    def get_rep_dim(self):
        """
        Get the dimension of the representation.

        Returns:
            int: dimension of the representation
        """
        return self.index

    def compute_rep(self, idx_to_perm):
        """
        Compute the natural representation by directly computing permutation matrices

        Args:
            idx_to_perm (function): Function that takes an index and returns the corresponding permutation object

        Returns:
            torch.tensor: (group.order, group.index, group.index) tensor of permutation matrices for each group element
        """
        idx = list(np.linspace(0, self.index-1, self.index))
        rep = torch.zeros(self.order, self.index, self.index).cuda()
        for x in range(self.order):
            rep[x, idx, idx_to_perm(x)(idx)] = 1
        return rep

class StandardRepresentation(Representation):
    """
    Generate the standard representation of the symmetric group.

    """
    def __init__(self, compute_rep_params, init_rep_params):
        """
        Initialise the standard representation. 

        Args:
            compute_rep_params (list): list containing natural_reps object necessary to calculate the standard representation
            init_rep_params (dict): standard group parameters needed by the representation, including index, order, multiplication_table, inverses, all_data
        """
        self.friendly_name = 'standard'
        super().__init__(compute_rep_params, **init_rep_params)

    def get_rep_dim(self):
        """
        Get the dimension of the representation.

        Returns:
            int: dimension of the representation
        """
        return self.index-1
    
    def compute_rep(self, natural_reps):
        """
        Compute the standard representation from the natural representation.

        Args:
            natural_reps (torch.tensor): (group.order, group.index, group.index) tensor of natural representations (permutation matrices)

        Returns:
            torch.tensor: (group.order, group.index-1, group.index-1) tensor of standard representations
        """
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
        
class SignRepresentation(Representation):
    """
    Initialise the sign representation of the symmetric group.

    """
    def __init__(self, compute_rep_params, init_rep_params):
        """
        Initialise the sign representation. 

        Args:
            compute_rep_params (list): list consisting of the signatures object required to compute the representation
            init_rep_params (dict): standard group parameters needed by the representation, including index, order, multiplication_table, inverses, all_data
        """
        self.friendly_name = 'sign'
        super().__init__(compute_rep_params, **init_rep_params)

    def get_rep_dim(self):
        """
        Get the dimension of the representation.

        Returns:
            int: dimension of the representation
        """
        return 1

    def compute_rep(self, signatures):
        """
        Compute the sign representation from the signatures.

        Args:
            signatures (torch.tensor): (group.order, 1) tensor of signatures

        Returns:
            torch.tensor: (group.order, 1, 1) tensor of sign representations
        """
        rep = torch.zeros(self.order, 1, 1).cuda()
        rep[:, 0, 0] = signatures
        return rep

class StandardSignRepresentation(Representation):
    def __init__(self, compute_rep_params, init_rep_params):
        """
        Initialise the tensor product of the standard and sign representation. 

        Args:
            compute_rep_params (list): list consisting of tensor of standard representation and signatures, required to compute the representation
            init_rep_params (dict): standard group parameters needed by the representation, including index, order, multiplication_table, inverses, all_data
        """
        self.friendly_name = 'standard_sign'
        super().__init__(compute_rep_params, **init_rep_params)

    def get_rep_dim(self):
        """
        Get the dimension of the representation.

        Returns:
            int: dimension of the representation
        """
        return self.index-1

    def compute_rep(self, standard_reps, signatures):
        """
        Compute the tensor product of the standard and sign representation.

        Args:
            standard_reps (torch.tensor): (group.order, group.index-1, group.index-1) tensor of standard representations
            signatures (torch.tensor): (group.order, 1) tensor of signatures

        Returns:
            torch.tensor: (group.order, group.index-1, group.index-1) tensor of standard_sign representations
        """
        rep = []
        for i in range(standard_reps.shape[0]):
            rep.append(signatures[i]*standard_reps[i])
        rep = torch.stack(rep, dim=0).cuda()
        return rep

class SymmetricRepresentationFromGenerators(Representation):
    def __init__(self, compute_rep_params, init_rep_params, name):
        """
        Initialise a representation of the symmetric group from a set of generators.

        Args:
            compute_rep_params (list): list consisting of the generators of the representation, sympy group object, and a function that maps indices to permutations
            init_rep_params (dict): standard group parameters needed by the representation, including index, order, multiplication_table, inverses, all_data
            name (str): name of the representation
        """
        self.friendly_name = name
        super().__init__(compute_rep_params, **init_rep_params)

    # TODO: make this less hacky
    def get_rep_dim(self):
        """
        Get the dimension of the representation.

        Returns:
            int: dimension of the representation
        """
        return list(self.compute_rep_params[0].values())[0].shape[0] # hacky way to get the dimension of the representation

    def compute_rep(self, generators, G, idx_to_perm):
        """
        Compute the representation from the generators.

        Args:
            generators (dict): dictionary of generators of the group along with their representations
            G (sympy group object): group object
            idx_to_perm (function): function that maps indices to permutations

        Returns:
            torch.tensor: (group.order, dim, dim) tensor of arbitrary representations
        """
        rep = torch.zeros(self.order, self.dim, self.dim).cuda()
        for i in range(self.order):
            generator_product = G.generator_product(idx_to_perm(i), original=True)
            result = torch.eye(self.dim).float()
            for g in generator_product:
                result = result @ generators[g]
            rep[i] = result
        return rep.cuda()

class Cyclic2dRepresentation(Representation):
    """
    Rotation matrix like representations of the cyclic group.
    """
    def __init__(self, compute_rep_params, init_rep_params, name):
        """
        Initialise a representation of the symmetric group from a set of generators.

        Args:
            compute_rep_params (list): list consisting of the k paramater of the representation
            init_rep_params (dict): standard group parameters needed by the representation, including index, order, multiplication_table, inverses, all_data
            name (str): name of the representation
        """
        self.friendly_name = name
        super().__init__(compute_rep_params, **init_rep_params)

    # TODO: make this less hacky
    def get_rep_dim(self):
        """
        Get the dimension of the representation.

        Returns:
            int: dimension of the representation
        """
        return 2 

    def compute_rep(self, k):
        rep = torch.zeros(self.order, 2, 2).cuda()
        for i in range(self.order):
            rep[i] = torch.tensor([[np.cos(2*np.pi*i*k/self.order), -np.sin(2*np.pi*i*k/self.order)], 
                                    [np.sin(2*np.pi*i*k/self.order), np.cos(2*np.pi*i*k/self.order)]]).cuda()
        return rep