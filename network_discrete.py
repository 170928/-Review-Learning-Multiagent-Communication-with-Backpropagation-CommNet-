import numpy as np
import tensorflow as tf

def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())
def bias_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())
def mu_variable(shape):
    return tf.Variable(tf.random_uniform(shape, minval = -tf.sqrt(3/shape[0]), maxval = tf.sqrt(3/shape[0])))
def sigma_variable(shape):
    return tf.Variable(tf.constant(0.017, shape = shape))

def noisy_dense(input_, input_shape, mu_w, sig_w, mu_b, sig_b, is_train_process):
    eps_w = tf.cond(is_train_process, lambda: tf.random_normal(input_shape), lambda: tf.zeros(input_shape))
    eps_b = tf.cond(is_train_process, lambda: tf.random_normal([input_shape[1]]), lambda: tf.zeros([input_shape[1]]))
    w_fc = tf.add(mu_w, tf.multiply(sig_w, eps_w))
    b_fc = tf.add(mu_b, tf.multiply(sig_b, eps_b))
    return tf.matmul(input_, w_fc) + b_fc


def CommNet(state, output_len, is_train):
    # state :: tf.placeholder(tf.float32, [None, S_DIM, AGENT_NUM], 'state')
    state_size = state.get_shape()[-2] # observation dimension
    agent_num = state.get_shape()[-1] # agent number

    # First layer
    f1_hidden_num = 512
    f1_hidden_layer = 3

    # state :: [None, S_DIM] x Num_agent
    state_unstack = tf.unstack(state, axis=-1)

    # weight 와 bias list를 만든다.
    f1_weight = [weight_variable('f1_w_h' + str(i), [f1_hidden_num, f1_hidden_num]) for i in range(f1_hidden_layer - 1)]
    f1_weight.insert(0, weight_variable('f1_w_input', [state_size, f1_hidden_num]))
    f1_bias = [weight_variable('f1_b_h' + str(i), [f1_hidden_num]) for i in range(f1_hidden_layer - 1)]
    f1_bias.insert(0, weight_variable('f1_b_input', [f1_hidden_num]))

    h_list = []
    for i in range(agent_num):
        temp_state = state_unstack[i]
        # for 문으로 list 내의 weight를 꺼내온다.
        for layer in range(f1_hidden_layer):
            if layer == 0:
                f1_hidden = tf.nn.relu(tf.matmul(temp_state, f1_weight[layer]) + f1_bias[layer])
            else:
                f1_hidden = tf.nn.relu(tf.matmul(f1_hidden, f1_weight[layer]) + f1_bias[layer])
        h_list.append(f1_hidden)

    # Communication Layer
    # h_stack : [None, hidden_size, num_agent)
    # comm = [None, hidden_size]
    h_stack = tf.stack(h_list, axis=2)
    comm = tf.reduce_mean(h_stack, axis=2)

    f_comm_hidden_num = 512

    f_comm_weight = weight_variable('f_comm_w_1', [f1_hidden_num, f_comm_hidden_num])
    c_comm_weight = weight_variable('c_comm_w_1', [f1_hidden_num, f_comm_hidden_num])

    comm_list = []
    for i in range(agent_num):
        temp_h = h_list[i]

        h_comm_hidden_state1 = tf.matmul(temp_h, f_comm_weight)
        c_comm_hidden_state1 = tf.matmul(comm, c_comm_weight)
        f_comm_hidden_state1 = tf.nn.tanh(h_comm_hidden_state1 + c_comm_hidden_state1)

        comm_list.append(f_comm_hidden_state1)

    ## Second Layer
    f2_hidden_num = 2 * 512
    f2_hidden_layer = 3

    f2_weight = [weight_variable('f2_w_h' + str(i), [f2_hidden_num, f2_hidden_num]) for i in range(f2_hidden_layer - 1)]
    f2_weight.insert(0, weight_variable('f2_w_input', [f2_hidden_num, f2_hidden_num]))
    f2_bias = [weight_variable('f2_b_h' + str(i), [f2_hidden_num]) for i in range(f2_hidden_layer - 1)]
    f2_bias.insert(0, weight_variable('f2_b_input', [f2_hidden_num]))

    output_weight_mu = mu_variable([f2_hidden_num, output_len])
    output_weight_sig = sigma_variable([f2_hidden_num, output_len])

    output_bias_mu = mu_variable([output_len])
    output_bias_sig = sigma_variable([output_len])

    # Second layer :: Skip connection is implemented here
    out_list = []
    for i in range(agent_num):
        # comm of first layer :: comm_list (?, 512)
        temp_comm = comm_list[i]
        # hidden of encoded :: h_list (?, 512)
        temp_h = h_list[i]
        # skip connedction (?, 1024)
        temp_input = tf.concat([temp_comm, temp_h], axis=1)

        for layer in range(f2_hidden_layer):
            if layer == 0:
                f2_hidden = tf.nn.relu(tf.matmul(temp_input, f2_weight[layer]) + f2_bias[layer])
            else:
                f2_hidden = tf.nn.relu(tf.matmul(f2_hidden, f2_weight[layer]) + f2_bias[layer])

        # output layer
        temp_out = noisy_dense(f2_hidden, [f2_hidden_num, output_len],
                               output_weight_mu, output_weight_sig,
                               output_bias_mu, output_bias_sig, is_train)
        out_list.append(temp_out)

    # output : (None, action_size, num_agent)
    output = tf.stack(out_list, axis=2)

    return output, out_list

if __name__ == "__main__":
    state = tf.placeholder(tf.float32, [None, 3, 4], 'state')
    CommNet(state, 4, tf.placeholder(tf.bool))