module multi_k_solver

export solver_multi_k, Info_upscale, print_status, save_values, update_K_p

include("./param.jl")

using Main.param
using ADCME
using SparseArrays

using DelimitedFiles
using Dates
using Plots
using Random

mutable struct tf_variables_definition
    # structrure with all the variables are optimized and placeholder
    lambda
    N_k_dis
    K_save
    K
    K_log_mean
    K_log_Var
end


function solver_multi_k(model_param)

    # model_param = param_model(N_k = 2, N_x = 10, N_y = 10, N_steps = 50)

    #Building matrices for applying BC
    ii_l = [i for i = 1:model_param.N_x:model_param.Ne]
    jj_l = [1 for i = 1:model_param.N_x:model_param.Ne]
    vv_l = [1.0 for i = 1:model_param.N_x:model_param.Ne]
    # B
    A_l_m = SparseTensor(ii_l, jj_l, vv_l, model_param.Ne, 1)

    ii_r = [i for i = model_param.N_x:model_param.N_x:model_param.Ne]
    jj_r = [1 for i = 1:model_param.N_x:model_param.Ne]
    vv_r = [1.0 for i = 1:model_param.N_x:model_param.Ne]
    A_r_m = SparseTensor(ii_r, jj_r, vv_r, model_param.Ne, 1)
    A_m = 1 - Array(A_l_m) - Array(A_r_m)

    x = LinRange(0, 1, model_param.N_x)


    list_m2_x = Int64[]
    list_m2_y = Int64[]
    list_m2_v = []

    list_m3_x = Int64[]
    list_m3_y = Int64[]

    list_m4_x = Int64[]
    list_m4_y = Int64[]

    list_m5_x = Int64[]
    list_m5_y = Int64[]

    for jj = 0:model_param.N_y-1
        for ii = 0:model_param.N_x-1

            if ii + 2 < model_param.N_x #&& ii>0
                append!(list_m2_x, jj * model_param.N_x + ii + 2)
                append!(list_m2_y, jj * model_param.N_x + ii + 1)
            end

            if ii > 1 #&& ii  < model_param.N_x-1
                append!(list_m3_x, jj * model_param.N_x + ii)
                append!(list_m3_y, jj * model_param.N_x + ii + 1)
            end

            if jj + 1 < model_param.N_y && ii > 0 && ii < model_param.N_x - 1
                append!(list_m4_x, (jj + 1) * model_param.N_x + ii + 1)
                append!(list_m4_y, jj * model_param.N_x + ii + 1)
            end

            if jj > 0 && ii < model_param.N_x - 1 && ii > 0
                append!(list_m5_x, (jj - 1) * model_param.N_x + ii + 1)
                append!(list_m5_y, jj * model_param.N_x + ii + 1)
            end


        end
    end

    list_m3_v_i = [ii + 1 for ii = 0:((model_param.N_x-1)*(model_param.N_y)-1) if ii % (model_param.N_x - 1) != 0]
    list_m2_v_i = [ii + 1 for ii = 0:((model_param.N_x-1)*(model_param.N_y)-1) if ii % (model_param.N_x - 1) != (model_param.N_x - 2)]
    list_m4_v_i = [ii + 1 for ii = 0:((model_param.N_x)*(model_param.N_y-1)-1) if ii % (model_param.N_x) != (model_param.N_x - 1) && ii % (model_param.N_x) != 0]



    function left_BC(t)

        Dh = 1e-1
        A = 1e-4
        B = 1
        H = B + (Dh * exp(-A * t))
    end

    function BC(t)

        BC_right_ = zeros(model_param.N_y)

        BC_left_ = ones(model_param.N_y) * constant(left_BC(t))
        #     BC_left_ = ones(N_y) + constant(t*1e-2)

        BC_left = SparseTensor(ii_l, jj_l, BC_left_, model_param.Ne, 1)
        BC_right = SparseTensor(ii_r, jj_r, BC_right_, model_param.Ne, 1)


        #     constant(ones(N_y) ) + constant(1e-4 * t )
        return BC_left, BC_right, BC_left_, BC_right_
    end

    function IC()

        _, _, BC_left, Bc_right = BC(0)

        h_0 = reshape((reshape(BC_left, (model_param.N_y, 1)) .- reshape(BC_left, (model_param.N_y, 1)) * reshape(constant(x), (1, model_param.N_x))), (1, model_param.Ne)) .* ones(model_param.N_k, 1)
        return h_0
    end

    function advance_time(h_rhs, i)
        h_n = TensorArray(model_param.N_k)
        j = constant(1, dtype = Int32)

        function condition(j, h_n)
            j <= N_k_dis
        end

        function body(j, h_n)

            K_avg_x_t = tf.squeeze(tf.slice(K_avg_x, constant([i, 0, 0, 0], dtype = Int32), constant([1, -1, -1, -1], dtype = Int32)))
            K_avg_y_t = tf.squeeze(tf.slice(K_avg_y, constant([i, 0, 0, 0], dtype = Int32), constant([1, -1, -1, -1], dtype = Int32)))


            K_avg_x_i = reshape(K_avg_x_t[i, j, :], -1) * model_param.dt / (model_param.dx)^2 / S
            K_avg_y_i = reshape(K_avg_y_t[i, j, :], -1) * model_param.dt / (model_param.dy)^2 / S

            m2_m = SparseTensor(list_m2_x, list_m2_y, K_avg_x_i[list_m2_v_i], model_param.Ne, model_param.Ne)
            m3_m = SparseTensor(list_m3_x, list_m3_y, K_avg_x_i[list_m3_v_i], model_param.Ne, model_param.Ne)
            m4_m = SparseTensor(list_m4_x, list_m4_y, K_avg_y_i[list_m4_v_i], model_param.Ne, model_param.Ne)
            m5_m = SparseTensor(list_m5_x, list_m5_y, K_avg_y_i[list_m4_v_i], model_param.Ne, model_param.Ne)


            m2_d = SparseTensor(list_m2_x, list_m2_x, K_avg_x_i[list_m2_v_i], model_param.Ne, model_param.Ne)
            m3_d = SparseTensor(list_m3_x, list_m3_x, K_avg_x_i[list_m3_v_i], model_param.Ne, model_param.Ne)
            m4_d = SparseTensor(list_m4_x, list_m4_x, K_avg_y_i[list_m4_v_i], model_param.Ne, model_param.Ne)
            m5_d = SparseTensor(list_m5_x, list_m5_x, K_avg_y_i[list_m4_v_i], model_param.Ne, model_param.Ne)


            A = spdiag(model_param.Ne) - (m2_m + m3_m + m4_m + m5_m) + (m2_d + m3_d + m4_d + m5_d)

            h_j = A \ h_rhs[j]

            # # update head

            h_n = write(h_n, j, h_j)

            # h_n = write(h_n, j, h_rhs[j])

            j + 1, h_n
        end

        _, out = while_loop(condition, body, [j, h_n])


        h_n = stack(out)

    end

    rng = MersenneTwister(1234)

    # K_log_mean = Variable(log.(ones(1, 1, 1) * 1e-5))
    # K_log_var = Variable((ones(1, 1, 1)))
    # K_log = ones(model_param.N_k, model_param.N_x, model_param.N_y) .* K_log_mean + randn(rng, (model_param.N_k, model_param.N_x, model_param.N_y)) .* K_log_var

    K_log_mean = Variable(log.(ones(model_param.N_steps, 1, 1, 1) * 1e-5))
    K_log_var = Variable((ones(model_param.N_steps, 1, 1, 1)))
    K_log = ones(1, model_param.N_k, model_param.N_x, model_param.N_y) .* K_log_mean + randn(rng, (1, model_param.N_k, model_param.N_x, model_param.N_y)) .* K_log_var

    K = tf.exp(K_log)

    K_save = tf.stack([K_log_mean, K_log_var])

    # K_log = Variable(log.(ones(model_param.N_k, 1, 1) * 1e-5))
    # K_log_field = K_log .* ones(1, model_param.N_x, model_param.N_y)
    # K = tf.exp(K_log)
    # K = K_exp .* ones(1, model_param.N_x, model_param.N_y)

    K_avg_x = 2 / (1 / tf.slice(K, constant([0, 0, 0, 0], dtype = Int32), constant([-1, -1, model_param.N_x - 1, -1], dtype = Int32)) + 1 / tf.slice(K, constant([0, 0, 1, 0], dtype = Int32), constant([-1, -1, model_param.N_x - 1, -1], dtype = Int32)))
    K_avg_y = 2 / (1 / tf.slice(K, constant([0, 0, 0, 0], dtype = Int32), constant([-1, -1, -1, model_param.N_y - 1], dtype = Int32)) + 1 / tf.slice(K, constant([0, 0, 0, 1], dtype = Int32), constant([-1, -1, -1, model_param.N_y - 1], dtype = Int32)))


    # K_avg_x = 2 / (1 / K[:, 2:end, :] + 1 / K[:, 1:end-1, :])
    # K_avg_y = 2 / (1 / K[:, :, 2:end] + 1 / K[:, :, 1:end-1])


    # Main function to solve the Darcy flow problem - forward problem
    N_k_dis = placeholder(model_param.N_k, dtype = Int32)
    lambda = placeholder(ones(1))
    # initialize the head with initial condition
    h_IC = IC()
    # Initialize the tensor of head for all time steps
    h_t = TensorArray(model_param.N_steps)


    # Initial state out of the Loop 
    # Initial condition

    # write the first head value in the tensor
    h_t = write(h_t, 1, constant(h_IC))
    # next step
    i = constant(2, dtype = Int32)


    S = 1e-6 # Storativity

    function condition(i, h_t_loop)
        # for loop until last time step
        i <= model_param.N_steps
    end

    function body(i, h_t_loop)

        # calculate time
        t = cast(i - 1, Float64) * model_param.dt

        # build rhs
        h_rhs = read(h_t_loop, i - 1)
        BC_left, BC_right = BC(t)
        h_rhs = h_rhs .* A_m[:, 1] + Array(BC_left)[:, 1] + Array(BC_right)[:, 1]

        # advance time
        h_next = advance_time(h_rhs, i)

        # # # update hydraulic head
        h_t_loop = write(h_t_loop, i, h_next)

        # return head and flow
        i + 1, h_t_loop

    end

    # # Define the tensor flow loop
    _, out = while_loop(condition, body, [i, h_t])

    h_t = stack(out)

    set_shape(h_t, (model_param.N_steps, model_param.N_k, model_param.Ne))

    i_qoi = model_param.N_x ÷ 2
    j_qoi = model_param.N_y ÷ 2
    ij_qoi = (j_qoi - 1 + 1) * model_param.N_x + i_qoi


    K_avg_x_ij = tf.squeeze(tf.slice(K_avg_x, constant([0, 0, i_qoi - 1, j_qoi - 1], dtype = Int32), constant([1, -1, 1, 1], dtype = Int32)))


    q_t_x = reshape(K_avg_x_ij, (model_param.N_steps, model_param.N_k)) .* (h_t[:, :, ij_qoi] - h_t[:, :, ij_qoi+1]) / model_param.dx


    tf_variables = tf_variables_definition(lambda, N_k_dis, K_save, K, K_log_mean, K_log_var)

    return tf_variables, h_t, q_t_x

end

function Info_upscale(tf_variables, model_param, q_t, N_k_dis_ = 2, maxiter = 400)

    function loss_function(lambda, p, y_t, q_t)
        # Definition of loss function for information theory upscale

        # q_t = tf.slice(q_t, constant([0, 0, 0], dtype = Int32), constant([-1, -1, 1], dtype = Int32))

        dw = tf.squared_difference(constant(reshape(transpose(y_t), (model_param.N_steps, model_param.N_k_fine, 1))), constant(reshape(q_t, (model_param.N_steps, 1, model_param.N_k))))

        # dw_2_sum = tf.reduce_mean(dw) * 1e5 # devide with momment

        dw_2_mean = tf.reduce_sum(tf.reduce_mean(dw, axis = 0) .* p) / model_param.N_k_fine

        momment2 = reshape(std(y_t, dims = 1), (model_param.N_steps, 1, 1)) .^ 2

        dw = tf.reduce_sum(dw ./ momment2, axis = 0)
        dw_2_sum = tf.reduce_mean(dw) * 1e5
        # dw = tf.reduce_sum(dw ,axis=0)

        dw_min = tf.reduce_min(dw, 1, keepdims = true)
        dw_m_min = -dw + dw_min

        F = p .* tf.exp(dw_m_min / lambda)

        F = tf.reduce_sum(F, 1, keep_dims = true)

        F = tf.log(F) - dw_min / lambda

        loss = -tf.reduce_sum(F) / model_param.N_k_fine

        return loss, dw_2_sum
    end

    files_q = string("./", "exitq_mean_50", ".txt")

    y_t = readdlm(files_q, ' ', Float64)

    y_t = y_t[:, 1:model_param.n_step_skip:end]

    # Define probabilities as parameters of softmax to assure sum(p)=1
    p_pre_soft_max_values = ones(1, model_param.N_k)# .+ (1e0 .* rand(1,model_param.N_k)) ; #CHANGED TO CHECK IF NOT OPTIMIZED --

    # Make only part of the probabilities to be active
    for i = 1:N_k_dis_
        p_pre_soft_max_values[1, i] += 100
    end

    # Pass values in to soft max to make them as probabilities
    p_pre_soft_max = Variable(p_pre_soft_max_values, trainable = false) #CHANGED TO CHECK IF NOT OPTIMIZED
    p = tf.nn.softmax(p_pre_soft_max, 1)

    loss, dw_2_sum = loss_function(tf_variables.lambda, p, y_t, q_t)


    sort_list = [tf.sort(tf.slice(q_t , [0, 0], [model_param.N_steps, tf_variables.N_k_dis]), axis = 1) ]

    # sort_list = [tf.sort(q_t, axis = 1)]
    diff_list = [tf.reduce_max(tf.reduce_min((tf.slice(sort_list[ii], [0, 1], [-1, -1]) -
                                              (tf.slice(sort_list[ii], [0, 0], [-1, tf_variables.N_k_dis - 1]))) ./ tf.reduce_mean(sort_list[ii], axis = 1, keep_dims = true), axis = 1)) for ii in 1]
    diff_eval = tf.reduce_max(tf.stack(diff_list))


    # Define all the optimization algorithm ADAM and LFGS for both MSE and information theory approach
    opt_ADAM = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss * 1e5)
    opt_LFGS = ScipyOptimizerInterface(loss * 1e5; method = "L-BFGS-B", options = Dict("maxiter" => maxiter * 2, "ftol" => 1e-14, "gtol" => 1e-14))
    opt_ADAM_sum = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(dw_2_sum)
    opt_LFGS_sum = ScipyOptimizerInterface(dw_2_sum * 1e5; method = "L-BFGS-B", options = Dict("maxiter" => maxiter, "ftol" => 1e-14, "gtol" => 1e-14))


    return loss, dw_2_sum, opt_ADAM, opt_LFGS, opt_ADAM_sum, opt_LFGS_sum, diff_eval, p_pre_soft_max, p
end


function save_values(sess, model_param, tf_variables, q_t, p, T_exp, mode = "a")
    # function to save values of permeabilities during optimization

    # name of current experiment
    exp_name = model_param.exp_name
    # evaluation of tensor flow variables and tensors
    k_x_save = run(sess, tf_variables.K_save)
    p_save = run(sess, p)
    q_x_save = run(sess, q_t)

    # Write in each txt the corresponding value
    open(string("./../results/", exp_name, "lambda.txt"), mode) do io
        writedlm(io, T_exp)
    end

    open(string("./../results/", exp_name, "k.txt"), mode) do io
        writedlm(io, k_x_save)
    end
    open(string("./../results/", exp_name, "p.txt"), mode) do io
        writedlm(io, p_save)
    end
    open(string("./../results/", exp_name, "q_x.txt"), mode) do io
        writedlm(io, q_x_save)
    end
end

function print_status(sess, model_param, loss, diff_eval, T_exp, T_, N_k_dis_, tf_variables, mode = "a")
    # Funtion to print the update for debuging and experiment evolution check

    exp_name = model_param.exp_name
    diff_, loss_ = run(sess, [diff_eval, loss], feed_dict = Dict(tf_variables.lambda => ones(1) * T_, tf_variables.N_k_dis => N_k_dis_))
    print("Saving Values at: ", Dates.format(now(), "HH:MM"), T_exp, "\t", T_, "\t", diff_, "\t", loss_, "\t", "\t", N_k_dis_, "\n")

end

function update_K_p(sess, model_param, tf_variables, check_diff, N_k_dis_, p_pre_soft_max)
    # function to update the permeabilities, if criteria are met the active probabilities are doubled otherwise the permeabilities are perturbed 
    # k_x_t_log = tf_variables.K_log
    k_log_mean = tf_variables.K_log_mean
    k_log_Var = tf_variables.K_log_Var
    N_k = model_param.N_k

    print("function for update K:", check_diff, "\t", N_k_dis_, "\n")
    if check_diff > 5e-9 && N_k_dis_ < N_k
    
        # k_x_t_update = run(sess, k_x_t_log)
        # k_x_t_update[N_k_dis_+1:N_k_dis_*2, :] = k_x_t_update[1:N_k_dis_, :]
        # k_x_t_update = k_x_t_update .+ (0.0 .+ 5e-3 .* (0.5 .- rand(N_k)))
        # run(sess, tf.assign(k_x_t_log, k_x_t_update))
    
    
        k_log_mean_update = run(sess, k_log_mean)
        k_log_mean_update[N_k_dis_+1:N_k_dis_*2, :] = k_log_mean_update[1:N_k_dis_, :]
        k_log_mean_update = k_log_mean_update .+ (0.0 .+ 5e-3 .* (0.5 .- rand(N_k)))
        run(sess, tf.assign(k_log_mean, k_log_mean_update))
    
        k_log_Var_update = run(sess, k_log_Var)
        k_log_Var_update[N_k_dis_+1:N_k_dis_*2, :] = k_log_Var_update[1:N_k_dis_, :]
        k_log_Var_update = k_log_Var_update .+ (0.0 .+ 5e-3 .* (0.5 .- rand(N_k)))
        run(sess, tf.assign(k_log_Var, k_log_Var_update))
    
    
    
        p_pre_soft_max_update = run(sess, p_pre_soft_max)
        p_pre_soft_max_update[:, N_k_dis_+1:N_k_dis_*2] = p_pre_soft_max_update[:, 1:N_k_dis_]
        p_pre_soft_max_update = p_pre_soft_max_update .* (1.0 .+ 0 * (rand(1, N_k) .- 0.5))
        run(sess, tf.assign(p_pre_soft_max, p_pre_soft_max_update))
    
        N_k_dis_ *= 2
    else
        # k_x_t_update = run(sess, k_x_t_log) .+ (0.0 .+ 5e-5 * (rand(N_k) .- 0.5))
        # run(sess, tf.assign(k_x_t_log, k_x_t_update))
    
        k_log_mean_update = run(sess, k_log_mean) .+ (0.0 .+ 5e-5 * (rand(N_k) .- 0.5))
        run(sess, tf.assign(k_log_mean, k_log_mean_update))
    
        k_log_Var_update = run(sess, k_log_Var) .+ (0.0 .+ 5e-5 * (rand(N_k) .- 0.5))
        run(sess, tf.assign(k_log_Var, k_log_Var_update))
    
        p_pre_soft_max_update = run(sess, p_pre_soft_max) .* (1 .+ 0 * (rand(1, N_k) .- 0.5))
        run(sess, tf.assign(p_pre_soft_max, p_pre_soft_max_update))
    end

    return N_k_dis_
end

end






