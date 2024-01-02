"""
    Simple experiment of PG-EXTRA combined with anchor acceleration:
        
    This code implements an experiment on the decentralized optimization problem,
    where PG-EXTRA and PG-EXTRA combined with anchor mechanism are applied.    
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



"""
    Helper functions: calculates inner product and norm induced by metric matrix M (which is dependent on alpha).

    (1) Minner  : calculates M-inner product between (x_1, w_1) and (x_2, w_2)
    (2) Mnormsq : calculates squared M-norm of (x, w)
    (3) Mnorm   : calculates M-norm of (x, w)
"""

def Minner(x1, w1, x2, w2, alpha, network_data, n_node) :
    Vred = network_data['Vred']
    Sred = network_data['Sred']

    norm_squared = (1/alpha)*np.sum(x1*x2) + (1/alpha)*np.sum(x1*w2) + (1/alpha)*np.sum(w1*x2)
    u1 = (1/alpha)*Vred.T@((Vred@w1)*np.reshape(1/np.sqrt(Sred), (n_node-1,1)))
    u2 = (1/alpha)*Vred.T@((Vred@w2)*np.reshape(1/np.sqrt(Sred), (n_node-1,1)))
    norm_squared = norm_squared + alpha*np.sum(u1*u2)

    return norm_squared

def Mnormsq(x, w, alpha, network_data, n_node) :
    return 1/4*Minner(x, w, x, w, alpha, network_data, n_node)

def Mnorm(x, w, alpha, network_data, n_node) :
    return 1/2*np.sqrt(Minner(x, w, x, w, alpha, network_data, n_node))



"""
    These functions generate the decentralized problem and solves it with PG-EXTRA and its combination with anchor acceleration.

    (1) data_generation : randomly generates problem data including
        - the measurement matrix A
        - (noisy) measurement b
        - true solution x_true, and its stacked version x_star
        - initial value x_0_data
    
    (2) graph_generation : generate the graph representing the network the decentralized optimization problem is to be solved.

    (3) obtain_opt_prob_solution : obtain true solution to the L1-regularized problem, which can be different from x_true due to noisy measurement.

    (4) pg_extra_with_anchor_acceleration : apply PG-EXTRA with or without anchor acceleration.
        - variable anchor gets a string as an input
              (i) anchor = None       : no anchor, (x_k, w_k) = T(x_{k-1}, w_{k-1}).
             (ii) anchor = "APPM"     : anchor 1/(k+1).
            (iii) anchor = "anchor"   : anchor = gamma / (k^p + gamma) where p, gamma are specified as additional input.
             (iv) anchor = "adaptive" : anchor = 1/2 * ||T(x_{k-1}, w_{k-1}) - (x_{k-1}, w_{k-1})||^2_M / ( ||T(x_{k-1}, w_{k-1}) - (x_{k-1}, w_{k-1})||^2_M + <T(x_{k-1}, w_{k-1}) - (x_{k-1}, w_{k-1}), (x_{k-1}, w_{k-1}) - (x_0, w_0)>_M ).
"""

def data_generation(problem_spec) :
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    n_nonzero_entry = problem_spec['n_nonzero_entry']
    
    # (1) sparse signal x_true
    x_true = np.zeros(vector_size)
    x_true[:n_nonzero_entry] = np.random.randn(n_nonzero_entry)
    np.random.shuffle(x_true)   # sparse signal x_true
    x_star = np.array(n_node*[x_true])

    # (2) stack of initial iterates x_0_data
    x_0_data = np.zeros((n_node, vector_size))
    x_0_data = np.random.randn(n_node, vector_size)

    # (3) measurement matrices
    A = np.random.randn(n_sensor, vector_size)
    for ii in range(n_node) :   # normalize each measurement matrices
        A[n_sensor_per_node*ii:n_sensor_per_node*(ii+1)] = A[n_sensor_per_node*ii:n_sensor_per_node*(ii+1)] / np.linalg.norm(A[n_sensor_per_node*ii:n_sensor_per_node*(ii+1)])
    b = A@x_true + 0.01 * np.random.randn(n_sensor)    # noisy measurements
    # b = A@x_true # exact measurement

    problem_data = {'x_true' : x_true, 'x_star' : x_star, 'x_0_data' : x_0_data, 'A' : A, 'b' : b}
    return problem_data


def graph_generation(problem_spec) :
    n_node = problem_spec['n_node']

    # (1) network
    G = nx.Graph()
    G.add_nodes_from([1, n_node])
    for node in range(1, n_node-1) :
        G.add_edges_from([(node, node+1)])
        G.add_edges_from([(node, node+2)])
    G.add_edges_from([(n_node-1, n_node)])

    # (2) mixing matrices : metropolis-hastings weights
    W = np.zeros((n_node, n_node))
    for edge in list(G.edges) :
        (i,j) = edge
        W[i-1,j-1] = 1/(np.maximum(G.degree(i), G.degree(j))+1)
        W[j-1,i-1] = 1/(np.maximum(G.degree(j), G.degree(i))+1)
    for i in range(n_node) :
        W[i,i] = 1 - np.sum(W[i])
    [_, S, V] = np.linalg.svd((1/2)*(np.eye(W.shape[0])-W))
    Vred = V[0:n_node-1]
    Sred = S[0:n_node-1]

    network_data = {'G' : G, 'W' : W, 'Vred' : Vred, 'Sred' : Sred}
    return network_data


def obtain_opt_prob_solution(problem_spec, problem_data, network_data) :
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    alpha = problem_data['alpha']
    rho = problem_data['rho']
    A = problem_data['A']
    b = problem_data['b']
    x_0_data = problem_data['x_0_data'] 
    itr_num = problem_data['itr_num']
    W = network_data['W']

    x_0 = np.array(x_0_data)
    x_k = np.array(x_0)
    w_k = np.zeros((n_node,vector_size))
    b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))

    for _ in range(2*itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        for jj in range(n_node) :
            A_temp = A[jj*n_sensor_per_node:(jj+1)*n_sensor_per_node]
            b_temp = b[jj*n_sensor_per_node:(jj+1)*n_sensor_per_node]
            x_k[jj] = W[jj]@x_k_prev - alpha*A_temp.T@(A_temp@x_k_prev[jj]-b_temp) - w_k_prev[jj]
            x_k[jj] = np.sign(x_k[jj])*np.maximum(np.abs(x_k[jj])-alpha*rho, 0.)
        w_k = w_k_prev + 1/2*(np.eye(n_node)-W)@x_k_prev
    x_opt_star = x_k
    f_star = 1/2*np.sum((A@x_opt_star.T-b_stack)**2) + rho*np.sum(np.abs(x_opt_star))

    return x_opt_star, f_star


def pg_extra_with_anchor_acceleration(anchor_type, problem_spec, problem_data, network_data, p=1.0, gamma=1.0) :
    n_sensor = problem_spec['n_sensor']
    n_sensor_per_node = problem_spec['n_sensor_per_node']
    n_node = problem_spec['n_node']
    vector_size = problem_spec['vector_size']
    alpha = problem_data['alpha']
    rho = problem_data['rho']
    A = problem_data['A']
    b = problem_data['b']
    x_0_data = problem_data['x_0_data']
    x_star = problem_data['x_star']
    itr_num = problem_data['itr_num']
    W = network_data['W']
    
    err_opt_star, err_star, op_norm, f_val, const_vio = [], [], [], [], []

    x_0 = np.array(x_0_data)
    x_k = np.array(x_0)
    w_0 = np.zeros((n_node,vector_size))
    w_k = np.array(w_0)
    b_stack = np.reshape(np.repeat(b, n_node), (n_sensor, n_node))

    for ii in range(itr_num) :
        x_k_prev = np.array(x_k)
        w_k_prev = np.array(w_k)
        for jj in range(n_node) :
            A_temp = A[jj*n_sensor_per_node:(jj+1)*n_sensor_per_node]
            b_temp = b[jj*n_sensor_per_node:(jj+1)*n_sensor_per_node]
            x_k[jj] = W[jj]@x_k_prev - alpha*A_temp.T@(A_temp@x_k_prev[jj]-b_temp) - w_k_prev[jj]
            x_k[jj] = np.sign(x_k[jj])*np.maximum(np.abs(x_k[jj])-alpha*rho, 0.)
        w_k = w_k_prev + 1/2*(np.eye(n_node)-W)@x_k_prev
        
        op_norm.append(Mnormsq(x_k-x_k_prev, w_k-w_k_prev, alpha, network_data, n_node))        # calculate operator norm

        # anchor acceleration
        beta_k = 0
        if anchor_type == "APPM" :
            beta_k = 1/(ii+1)
        elif anchor_type == "anchor" :
            beta_k = gamma / (ii**p+gamma)
        elif anchor_type == "adaptive" :
            op_normnormsq = Mnormsq(x_k_prev - x_k, w_k_prev - w_k, alpha, network_data, n_node)
            innerpd = Minner(x_k_prev - x_k, w_k_prev - w_k, x_0 - x_k_prev, w_0 - w_k_prev, alpha, network_data, n_node)
            beta_k = (1/2) * op_normnormsq / (op_normnormsq + innerpd)
        
        x_k = (1-beta_k) * x_k + (beta_k) * x_0
        w_k = (1-beta_k) * w_k + (beta_k) * w_0

        err_opt_star.append(np.sqrt(np.sum((x_k-x_opt_star)**2)))
        err_star.append(np.sqrt(np.sum((x_k-x_star)**2)))
        const_vio.append(np.sum((A@x_k.T-b_stack)**2))
        f_val.append(1/2*np.sum((A@x_k.T-b_stack)**2) + rho*np.sum(np.abs(x_k)))

    return op_norm, err_opt_star, err_star, const_vio, f_val




"""
    Implementation of the experiment in the paper.
"""
if __name__ == "__main__" :
    # random seed
    np.random.seed(108)

    # data generation
    problem_spec = {}
    problem_spec['n_sensor'] = 80
    problem_spec['n_sensor_per_node'] = 4
    problem_spec['n_node'] = 20
    problem_spec['vector_size'] = 100
    problem_spec['n_nonzero_entry'] = 0
    problem_data = data_generation(problem_spec)
    network_data = graph_generation(problem_spec)
    p, gamma = 1.5, 2.0

    # PG-EXTRA hyperparameters
    problem_data['alpha'] = 0.01
    problem_data['rho'] = 0.01
    problem_data['itr_num'] = 10000

    # obtain true solution to L1-regularized optimization problem
    x_opt_star, f_star = obtain_opt_prob_solution(problem_spec, problem_data, network_data)

    # apply PG-EXTRA combined with anchor acceleration
    op_norm, err_opt_star, err_star, const_vio, f_val = pg_extra_with_anchor_acceleration(None, problem_spec, problem_data, network_data)
    op_norm_APPM, err_opt_star_APPM, err_star_APPM, const_vio_APPM, f_val_APPM = pg_extra_with_anchor_acceleration("APPM", problem_spec, problem_data, network_data)
    op_norm_anchor, err_opt_star_anchor, err_star_anchor, const_vio_anchor, f_val_anchor = pg_extra_with_anchor_acceleration("anchor", problem_spec, problem_data, network_data, p, gamma)
    op_norm_adaptive, err_opt_star_adaptive, err_star_adaptive, const_vio_adaptive, f_val_adaptive = pg_extra_with_anchor_acceleration("adaptive", problem_spec, problem_data, network_data)

    # plot the result and save figures
    import os
    os.makedirs('./plots/', exist_ok=True)

    plt.rcdefaults()
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["legend.framealpha"] = 0.0
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["font.family"] = "sans" # "sans" "sans-serif"
    plt.rcParams["mathtext.fontset"] = 'cm' # default = 'dejavusans', other options = 'cm', 'stixsans', 'dejavuserif'
    black, red, blue, green = 'dimgrey', 'coral', 'deepskyblue', 'gold'

    # network graph plot
    plt.figure(figsize=(5,4))
    G = network_data['G']
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, edge_color='black',  with_labels=True, font_weight='light', node_size= 300, node_color='dimgray', width= 0.9, font_size=8)
    nx.draw_networkx_nodes(G, pos, nodelist = np.arange(1,21), node_color = 'lightgray', node_size = 200)
    plt.savefig(f'./plots/pg-extra-graph.pdf', dpi=300)
    plt.show()

    # operator norm plot (log-log plot)
    plt.figure(figsize=(5,4))
    plt.minorticks_off()
    plt.suptitle(r"anchor $\frac{\gamma}{k^p+\gamma}$ " + f"with p={p}, $\gamma$={gamma}")
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(op_norm, label='PG-EXTRA', color=black, linewidth=3)
    plt.plot(op_norm_APPM, label='with APPM', color=blue, linewidth=3)
    plt.plot(op_norm_anchor, label=r'with anchor, $\frac{\gamma}{k^p+\gamma}$', color=green, linewidth=3.5)
    plt.plot(op_norm_adaptive, label='with Adaptive', color=red, linewidth=2, linestyle='--')
    plt.xlabel(r"Iteration count $k$")
    plt.ylabel(r"$\|\tilde{\mathbf{A}}x_k\|^2_M$")
    plt.ylim(1e-10, 1e6)
    plt.legend()
    plt.savefig(f'./plots/pg-extra-op_norm-loglog.pdf', dpi=300)
    plt.show()

    # operator norm plot (log plot)
    plt.figure(figsize=(5,4))
    plt.suptitle(r"anchor $\frac{\gamma}{k^p+\gamma}$ " + f"with p={p}, $\gamma$={gamma}")
    plt.yscale("log")
    plt.plot(op_norm, label='PG-EXTRA', color=black, linewidth=3)
    plt.plot(op_norm_APPM, label='with APPM', color=blue, linewidth=3)
    plt.plot(op_norm_anchor, label=r'with anchor, $\frac{\gamma}{k^p+\gamma}$', color=green, linewidth=3.5)
    plt.plot(op_norm_adaptive, label='with Adaptive', color=red, linewidth=2, linestyle='--')
    plt.xlabel(r"Iteration count $k$")
    plt.ylabel(r"$\|\tilde{\mathbf{A}}x_k\|^2_M$")
    plt.ylim(1e-8, 1e6)
    plt.legend()
    plt.savefig(f'./plots/pg-extra-op_norm.pdf', dpi=300)
    plt.show()