include("./param.jl");
include("./aux_functions.jl");

using Main.aux_functions
using Main.param
using ADCME
using SparseArrays

using DelimitedFiles
using Dates
using Plots

maxiter = parse(Int64,ARGS[1])
N_steps = parse(Int64,ARGS[2])
ADAM = parse(Int64,ARGS[3])
print("maximum iteration: ",maxiter,"\t","Number of steps: ",N_steps,"\n")
param_model_val = param_model(N_steps=N_steps);
tf_variables, h_t, q_t_x, q_t_y = Darcy_flow_solver(param_model_val);

loss, opt_ADAM, opt_LFGS, opt_ADAM_sum, opt_LFGS_sum, diff_eval,p_pre_soft_max, p = Info_upscale(tf_variables,param_model_val,q_t_x, q_t_y,maxiter)

sess = Session(); init(sess);


global N_k_dis_ = 4
global T_exp = -2
T_=  10.0 .^ -T_exp

save_values(sess,param_model_val,tf_variables,q_t_x, q_t_y,p,"w")

print_status(sess,loss,diff_eval,T_exp,T_,N_k_dis_,tf_variables)

if ADAM == 1
    print("ADAM:", ADAM)
    for i = 1:10000
        
        run(sess, opt_ADAM_sum,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>4))

    end
else
    print("LFGS:", ADAM)
    ScipyOptimizerMinimize(sess, opt_LFGS_sum,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>4))
end
print_status(sess,loss,diff_eval,T_exp,T_,N_k_dis_,tf_variables)

check_diff = run(sess,diff_eval,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))

T_exp_final =4


while T_exp <= T_exp_final

    global T_=  10.0 ^ (-T_exp) 

    print_status(sess,loss,diff_eval,T_exp,T_,N_k_dis_,tf_variables)
    check_diff_ = run(sess,diff_eval,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))  
    print_status(sess,loss,diff_eval,T_exp,T_,N_k_dis_,tf_variables)
    ScipyOptimizerMinimize(sess, opt_LFGS,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))
    print_status(sess,loss,diff_eval,T_exp,T_,N_k_dis_,tf_variables)
    if round(T_exp,digits=2)%1 == 0

        save_values(sess,param_model_val,tf_variables,q_t_x, q_t_y,p)
        check_diff_ = run(sess,diff_eval,feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))  
        global N_k_dis_ = update_K_p(sess,param_model_val,tf_variables,check_diff_,N_k_dis_,p_pre_soft_max)

    end


    global T_exp += 0.1
end
