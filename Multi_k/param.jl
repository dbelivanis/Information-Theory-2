# Definition of module param that encapsulates all the parameters of the run of the optimization

module param
()
export param_model, Multi_point_flux_aux_matrix, load_QoIs
using Parameters, DelimitedFiles, ADCME, SparseArrays, Dates

@with_kw struct param_model
    # main structure with the most important parameters

    exp_name = string("experiment", Dates.format(now(), "dd_mm_HH_MM_S"))
    run_name = "exitq_mean_50"
    N_points = 9

    Lx::Int64 = 1000
    Ly::Int64 = 1000
    T_h::Int64 = 54 * 1600
    T::Int64 = T_h

    N_steps_orig::Int64 = 1600
    N_steps::Int64 = 100
    N_k::Int64 = 64

    ## Fine scale Inputs
    N_x::Int64 = 21
    N_y::Int64 = 21
    Ne::Int64 = N_x * N_y
    N_steps_fine::Int64 = N_steps
    N_k_fine::Int64 = 100
    # N_y_fine::Int64 = 1;

    n_step_skip::Int64 = N_steps_orig ÷ N_steps

    dt::Float64 = T_h / N_steps
    dx::Float64 = Lx / (N_x - 1)
    dy::Float64 = Ly / (N_y - 1)

    # loc_x_list = [6, 11, 16, 6, 11, 16, 6, 11, 16]
    # loc_y_list = [6, 6, 6, 11, 11, 11, 16, 16, 16]

    # K_mupltiplier::Float64 = 1
end

# function load_QoIs(model_param)
#     # function to load the quantity of interest and calculate the standard deviation

#     files_x = [string("./", model_param.run_name, "/Q_x_", string(ii), ".txt") for ii = 1:9]
#     files_y = [string("./", model_param.run_name, "/Q_y_", string(ii), ".txt") for ii = 1:9]

#     y_x_list = [readdlm(files_x[ii], ' ', Float64) for ii = 1:9]
#     y_y_list = [readdlm(files_y[ii], ' ', Float64) for ii = 1:9]

#     momment2 = maximum([maximum(std(y_x_list)), maximum(std(y_y_list))])
#     n_step_skip = model_param.N_steps_orig ÷ model_param.N_steps

#     y_x_list = [y_x_list[ii][:, 1:n_step_skip:1800] for ii = 1:9]# automate the time steps
#     y_y_list = [y_y_list[ii][:, 1:n_step_skip:1800] for ii = 1:9]

#     return momment2, y_x_list, y_y_list
# end
end


