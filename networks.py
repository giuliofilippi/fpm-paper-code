# Imports
import torch
import torch.nn as nn

# Holistic Memory Network Class
class MushroomBodyNetwork(nn.Module):
    # init
    def __init__(self, 
                 n_vpn, 
                 n_kc, 
                 k, 
                 p, 
                 alpha=0.99, 
                 mode="top", 
                 init_weights=True, 
                 weight_mode='1', 
                 training_mode='exp',
                 epsilon=0):
        # init
        super(MushroomBodyNetwork, self).__init__()
        self.n_vpn = n_vpn
        self.n_kc = n_kc
        self.k = k
        self.p = p
        self.alpha = alpha
        self.mode = mode
        self.num_act = int(n_kc * p)
        self.weight_mode = weight_mode
        self.training_mode = training_mode
        self.epsilon = epsilon
        
        # Define layers
        self.mb_layer = nn.Linear(self.n_vpn, self.n_kc, bias=False)
        self.mbon_layer = nn.Linear(self.n_kc, 1, bias=False)

        # Only initialize if init_weights is True
        if init_weights:
            # Initialize hidden weights
            self.initialize_connections()

            # Initialize output weights
            self.initialize_output_weights()
        
    def initialize_connections(self):
        # Initialize connection weights between MB layer and VPNS
        weights = torch.zeros(self.n_kc, self.n_vpn)
        
        # Randomly choose connected indices for each row
        connected_indices = torch.randint(0, self.n_vpn, (self.n_kc, self.k))
        
        if self.weight_mode == '1':
            # Set connection weights to 1/k for selected indices
            weights.scatter_(1, connected_indices, 1/self.k)
        
        else:
            # Set connection weights to random value for selected indices
            random_weights = torch.randn(self.n_kc, self.n_vpn)
            weights.scatter_(1, connected_indices, random_weights)

        # Store weights as sparse matrix
        self.mb_layer.weight = torch.nn.Parameter(weights.to_sparse())

    def initialize_output_weights(self):
        # Initialize output weights to 1
        self.mbon_layer.weight.data.fill_(1/self.num_act)

    def initialize_rnd_output_weights(self):
        # Initialize output weights with random values in the range [0, 2/self.num_act]
        self.mbon_layer.weight.data = torch.rand(self.mbon_layer.weight.data.size()) * (2 / self.num_act)
        
    def inhibit_mb_layer(self, mb_output):
        # Apply inhibition to Mushroom Body layer
        num_neurons_to_keep = int(self.p * self.n_kc)
        _, indices = torch.topk(mb_output, num_neurons_to_keep, dim=1)
        mb_output_zeroed = torch.zeros_like(mb_output)
        mb_output_zeroed.scatter_(1, indices, 1)
        return mb_output_zeroed
    
    def threshold_mb_layer(self, mb_output):
        # Apply thresholding to Mushroom Body layer
        threshold = 1 - self.p
        zero_vec = torch.zeros_like(mb_output)
        one_vec = torch.ones_like(mb_output)
        mb_output_thresh = torch.where(mb_output >= threshold, one_vec, zero_vec)
        return mb_output_thresh
    
    def forward(self, x):
        # Flatten input
        x = x.reshape(-1, self.n_vpn)

        if self.weight_mode != '1':
            x = 2*x - 1
        
        # Forward pass through Mushroom Body layer
        mb_output = self.mb_layer(x)

        # noise
        if self.epsilon != 0:
            noise = torch.randn_like(mb_output) * self.epsilon
            mb_output += noise

        # nonlinearity in MB layer
        if self.mode == 'top':
            mb_output = self.inhibit_mb_layer(mb_output)
        elif self.mode == 'thresh':
            mb_output = self.threshold_mb_layer(mb_output)
        else:
            raise ValueError("mode should be top or thresh")
        
        # Forward pass through MBON layer
        mbon_output = self.mbon_layer(mb_output)
        
        # return outputs of both layers
        return mb_output, mbon_output

    def train_network(self, image_sequence):
        # convert to correct format
        if isinstance(image_sequence, list):
            if isinstance(image_sequence[0], torch.Tensor):
                image_sequence = torch.stack(image_sequence).unsqueeze(1)
            else:
                image_sequence = torch.Tensor(image_sequence).unsqueeze(1)
        else:
            image_sequence = torch.Tensor(image_sequence).unsqueeze(1)

        # forward pass
        mb_outputs = self.forward(image_sequence)[0]
        kc_counts = torch.sum(mb_outputs, dim=0)

        if self.training_mode == 'exp':
            # exponential weight update
            mult_factors = self.alpha**kc_counts
            self.mbon_layer.weight.data[0,:]*= mult_factors
        elif self.training_mode == 'additive':
            # additive weight update
            beta = 1/image_sequence.shape[0]
            additive_factors = beta*kc_counts
            self.mbon_layer.weight.data[0,:]-= additive_factors
        else:
            # binary weight update
            mult_factors = self.alpha**(kc_counts>0)
            self.mbon_layer.weight.data[0,:]*= mult_factors

        # return kc activations
        return mb_outputs
    
    def save_weights(self, file_path):
        # Save the weights of mb_layer as a sparse matrix
        torch.save(self.mb_layer.weight, file_path)

    def load_weights(self, file_path):
        # Load the weights from file_path into mb_layer
        self.mb_layer.weight = nn.Parameter(torch.load(file_path))
    
# Weighed Holistic Memory Network Class
class WeighedMBNetwork(nn.Module):
    # init
    def __init__(self, 
                 n_vpn, 
                 n_kc, 
                 k, 
                 p, 
                 q, 
                 alpha=0.99, 
                 mode="top", 
                 init_weights=True, 
                 weight_mode='1', 
                 training_mode='exp',
                 epsilon=0):
        # init
        super(WeighedMBNetwork, self).__init__()
        self.n_vpn = n_vpn
        self.n_kc = n_kc
        self.k = k
        self.p = p
        self.q = q
        self.alpha = alpha
        self.mode = mode
        self.num_act = int(n_kc * p)
        self.weight_mode = weight_mode
        self.training_mode = training_mode
        self.epsilon = epsilon
        
        # Define layers
        self.mb_layer = nn.Linear(self.n_vpn, self.n_kc, bias=False)
        self.mbon_layer = nn.Linear(self.n_kc, 1, bias=False)
        
        # Only initialize if init_weights is True
        if init_weights:
            # define q matrix for sampling
            self.q_matrix = torch.tensor([self.q]*self.n_kc, dtype=float)

            # Initialize hidden weights
            self.initialize_connections()

            # Initialize output weights
            self.initialize_output_weights()
        
    def initialize_connections(self):
        # Initialize connection weights between input and MB layers
        weights = torch.zeros(self.n_kc, self.n_vpn)
        
        # Randomly choose connected indices for each row
        connected_indices = torch.multinomial(self.q_matrix, self.k)
        
        if self.weight_mode == '1':
            # Set connection weights to 1/k for selected indices
            weights.scatter_(1, connected_indices, 1/self.k)
        
        else:
            # Set connection weights to random value for selected indices
            random_weights = torch.randn(self.n_kc, self.n_vpn)
            weights.scatter_(1, connected_indices, random_weights)
        
        # Store weights as sparse matrix
        self.mb_layer.weight = torch.nn.Parameter(weights.to_sparse())

    def initialize_output_weights(self):
        # Initialize output weights to 1
        self.mbon_layer.weight.data.fill_(1/self.num_act)

    def initialize_rnd_output_weights(self):
        # Initialize output weights with random values in the range [0, 2/self.num_act]
        self.mbon_layer.weight.data = torch.rand(self.mbon_layer.weight.data.size()) * (2 / self.num_act)
        
    def inhibit_mb_layer(self, mb_output):
        # Apply inhibition to Mushroom Body layer
        num_neurons_to_keep = int(self.p * self.n_kc)
        _, indices = torch.topk(mb_output, num_neurons_to_keep, dim=1)
        mb_output_zeroed = torch.zeros_like(mb_output)
        mb_output_zeroed.scatter_(1, indices, 1)
        return mb_output_zeroed
    
    def threshold_mb_layer(self, mb_output):
        # Apply thresholding to Mushroom Body layer
        threshold = 1 - self.p
        zero_vec = torch.zeros_like(mb_output)
        one_vec = torch.ones_like(mb_output)
        mb_output_thresh = torch.where(mb_output >= threshold, one_vec, zero_vec)
        return mb_output_thresh
    
    def forward(self, x):
        # Flatten input
        x = x.reshape(-1, self.n_vpn)

        # change if weight mode not 1
        if self.weight_mode != '1':
            x = 2*x - 1 
        
        # Forward pass through Mushroom Body layer
        mb_output = self.mb_layer(x)

        # noise
        if self.epsilon != 0:
            noise = torch.randn_like(mb_output) * self.epsilon
            mb_output += noise

        # nonlinearity in MB layer
        if self.mode == 'top':
            mb_output = self.inhibit_mb_layer(mb_output)
        elif self.mode == 'thresh':
            mb_output = self.threshold_mb_layer(mb_output)
        else:
            raise ValueError("mode should be top or thresh")
        
        # Forward pass through MBON layer
        mbon_output = self.mbon_layer(mb_output)
        
        # return outputs of both layers
        return mb_output, mbon_output

    def train_network(self, image_sequence):
        # convert to correct format
        if isinstance(image_sequence, list):
            if isinstance(image_sequence[0], torch.Tensor):
                image_sequence = torch.stack(image_sequence).unsqueeze(1)
            else:
                image_sequence = torch.Tensor(image_sequence).unsqueeze(1)
        else:
            image_sequence = torch.Tensor(image_sequence).unsqueeze(1)

        # forward pass
        mb_outputs = self.forward(image_sequence)[0]
        kc_counts = torch.sum(mb_outputs, dim=0)

        if self.training_mode == 'exp':
            # exponential weight update
            mult_factors = self.alpha**kc_counts
            self.mbon_layer.weight.data[0,:]*= mult_factors
        elif self.training_mode == 'additive':
            # additive weight update
            beta = 1/image_sequence.shape[0]
            additive_factors = beta*kc_counts
            self.mbon_layer.weight.data[0,:]-= additive_factors
        else:
            # binary weight update
            mult_factors = self.alpha**(kc_counts>0)
            self.mbon_layer.weight.data[0,:]*= mult_factors

        # return kc activations
        return mb_outputs
    
    def save_weights(self, file_path):
        # Save the weights of mb_layer as a sparse matrix
        torch.save(self.mb_layer.weight, file_path)

    def load_weights(self, file_path):
        # Load the weights from file_path into mb_layer
        self.mb_layer.weight = nn.Parameter(torch.load(file_path))
    
# Left/Right MB Network
class LRMBNetwork(nn.Module):
    # init
    def __init__(self, 
                 n_vpn, 
                 n_kc, 
                 k, 
                 p, 
                 q_l, 
                 q_r, 
                 alpha=0.99, 
                 mode="top", 
                 init_weights=True, 
                 weight_mode='1', 
                 training_mode='exp',
                 epsilon=0):
        # init
        super(LRMBNetwork, self).__init__()
        self.n_vpn = n_vpn
        self.n_kc = n_kc
        self.k = k
        self.p = p
        self.q_left = q_l
        self.q_right = q_r
        self.alpha = alpha
        self.mode = mode
        self.weight_mode = weight_mode
        self.training_mode = training_mode
        self.epsilon = epsilon
        
        # initiate L/R Networks
        self.LeftMB = WeighedMBNetwork(n_vpn, n_kc, k, p, q_l, alpha=alpha, mode=mode, init_weights=init_weights, weight_mode=weight_mode, training_mode=training_mode, epsilon=epsilon)
        self.RightMB = WeighedMBNetwork(n_vpn, n_kc, k, p, q_r, alpha=alpha, mode=mode, init_weights=init_weights, weight_mode=weight_mode, training_mode=training_mode, epsilon=epsilon)

    def forward(self, input):
        # Forward pass through left Mushroom Body
        _, left_mbon_output = self.LeftMB.forward(input)
        
        # Forward pass through right Mushroom Body
        _, right_mbon_output = self.RightMB.forward(input)

        # Sum Mbon outputs
        sum_mbon_output = left_mbon_output + right_mbon_output

        # Difference Mbon outputs
        diff_mbon_output = left_mbon_output - right_mbon_output
        
        # Return outputs of both left and right Mushroom Body networks
        return left_mbon_output, right_mbon_output, sum_mbon_output, diff_mbon_output

    def train_network(self, image_sequence):
        # Train Left MB
        left_mb_output = self.LeftMB.train_network(image_sequence)

        # Train Right MB
        right_mb_output = self.RightMB.train_network(image_sequence)
        
        # Combine outputs from both sides
        combined_mb_output = torch.cat((left_mb_output, right_mb_output), dim=1)
        
        # Return combined Mushroom Body outputs
        return combined_mb_output