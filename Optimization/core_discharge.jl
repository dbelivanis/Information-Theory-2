include("./param_discharge.jl");
include("./aux_functions_discharge.jl");

using Main.aux_functions
using Main.param
using ADCME
using SparseArrays

using DelimitedFiles
using Dates
using Plots

# definition of the parameters from the shell file
maxiter = parse(Int64, ARGS[1])
N_steps = parse(Int64, ARGS[2])
print("maximum iteration: ", maxiter, "\t", "Number of steps: ", N_steps, "\n")

# Definition of the parameters of initial descrite probabilities and initial lambda
global N_k_dis_ = 2
global T_exp = -3.0

# Initialization of the model and the optimization process
param_model_val = param_model(N_steps = N_steps);
tf_variables, h_t, q_t_x, q_t_y = Darcy_flow_solver(param_model_val);
loss, dw_2, opt_ADAM, opt_LFGS, opt_ADAM_sum, opt_LFGS_sum, diff_eval, p_pre_soft_max, p, opt_LFGS_x = Info_upscale(tf_variables, param_model_val, q_t_x, q_t_y, N_k_dis_, maxiter)

# Initialization of the session
sess = Session();
init(sess);


# Initial lambda
T_ = 10.0 .^ -T_exp


# Save values of the initial guess
save_values(sess, param_model_val, tf_variables, q_t_x,  p, T_exp, "w")
print_status(sess, param_model_val, loss, diff_eval, T_exp, T_, N_k_dis_, tf_variables, "w")

# Initial optimization with mean value as target
ScipyOptimizerMinimize(sess, opt_LFGS_sum, feed_dict = Dict(tf_variables.lambda => ones(1) * T_, tf_variables.N_k_dis => 64))
print("first bfgs")
save_values(sess, param_model_val, tf_variables, q_t_x,  p, T_exp)
print_status(sess, param_model_val, loss, diff_eval, T_exp, T_, N_k_dis_, tf_variables)
check_diff = run(sess, diff_eval, feed_dict = Dict(tf_variables.lambda => ones(1) * T_, tf_variables.N_k_dis => N_k_dis_))

# Definition of the final lambda
T_exp_final = 15

# Deterministic annealing loop ##
while T_exp <= T_exp_final

    # save the values before optimization
    if round(T_exp, digits = 2) * 10 % 1 == 0
        save_values(sess, param_model_val, tf_variables, q_t_x,  p, T_exp)
    end

    # Definition of the lambda value
    global T_ = 10.0^(-T_exp)

    # print_status(sess,param_model_val,loss,diff_eval,T_exp,T_,N_k_dis_,tf_variables)
    # check_diff_ = run(sess,diff_eval,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))  

    # Print status before optimization
    print("pre BFGS")
    print_status(sess, param_model_val, loss, diff_eval, T_exp, T_, N_k_dis_, tf_variables)

    # Run optimization
    # ScipyOptimizerMinimize(sess, opt_LFGS_x,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))
    # ScipyOptimizerMinimize(sess, opt_LFGS_y,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))

    ScipyOptimizerMinimize(sess, opt_LFGS, feed_dict = Dict(tf_variables.lambda => ones(1) * T_, tf_variables.N_k_dis => N_k_dis_))

    # Print status post optimization
    print("post BFGS")
    print_status(sess, param_model_val, loss, diff_eval, T_exp, T_, N_k_dis_, tf_variables)

    # Save values post optimization every 10 optimization processes and check for the discritization of the cdf to increase the descrite probabilities
    if round(T_exp, digits = 2) * 10 % 1 == 0
        save_values(sess, param_model_val, tf_variables, q_t_x,  p, T_exp)
        check_diff_ = run(sess, diff_eval, feed_dict = Dict(tf_variables.lambda => ones(1) * T_, tf_variables.N_k_dis => N_k_dis_))
        global N_k_dis_ = update_K_p(sess, param_model_val, tf_variables, check_diff_, N_k_dis_, p_pre_soft_max)
    end

    # Increase the lambda to continue the loop
    global T_exp += 0.1
end