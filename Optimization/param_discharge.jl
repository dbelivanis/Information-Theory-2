# Definition of module param that encapsulates all the parameters of the run of the optimization

module param
()
export param_model, Multi_point_flux_aux_matrix, load_QoIs
using Parameters, DelimitedFiles, ADCME, SparseArrays, Dates

@with_kw struct param_model
    # main structure with the most important parameters

    exp_name = string("experiment", Dates.format(now(), "dd_mm_HH_MM_S"))
    run_name = "Run_no_zero_flow_no_oscilations_Run_Run_1646968123"
    N_points = 9

    Lx::Int64 = 1000
    Ly::Int64 = 1000
    T_h::Int64 = 18000
    T::Int64 = T_h

    N_steps_orig::Int64 = 1800
    N_steps::Int64 = 100
    N_k::Int64 = 64

    ## Fine scale Inputs
    N_x::Int64 = 20
    N_y::Int64 = 20
    Ne::Int64 = N_x * N_y
    N_steps_fine::Int64 = N_steps
    N_k_fine::Int64 = 100
    # N_y_fine::Int64 = 1;

    n_step_skip::Int64 = N_steps_orig ÷ N_steps

    dt::Float64 = T_h / N_steps
    dx::Float64 = Lx / (N_x - 1)
    dy::Float64 = Ly / (N_y - 1)

    loc_x_list = [6, 11, 16, 6, 11, 16, 6, 11, 16]
    loc_y_list = [6, 6, 6, 11, 11, 11, 16, 16, 16]

    K_mupltiplier::Float64 = 1
end

struct aux_matrix
    # Structure to encapsulate the matrix for the linear solver
    ii_l
    jj_l
    vv_l
    ii_r
    jj_r
    vv_r
    m2_m
    m3_m
    m4_m
    m5_m
    m2_m_1
    m3_m_1
    m4_m_1
    m5_m_1
    A_m
end

function Multi_point_flux_aux_matrix(param)
    # function to create the martix for the multipoint flux approximation

    # loc_x_list = [6,11,16, 6,11,16, 6,11,16]
    # loc_y_list = [6, 6, 6,11,11,11,16,16,16];

    list_m2_x = Int64[]
    list_m2_y = Int64[]
    list_m2_v = Int64[]

    list_m3_x = Int64[]
    list_m3_y = Int64[]

    list_m4_x = Int64[]
    list_m4_y = Int64[]

    list_m5_x = Int64[]
    list_m5_y = Int64[]

    for jj in 0:param.N_y-1
        for ii in 0:param.N_x-1
            if ii != 0 && ii != param.N_x - 1
                if ii + 1 < param.N_x
                    append!(list_m2_x, Int(jj * param.N_x + ii + 1))
                    append!(list_m2_y, (jj) * param.N_x + (ii + 1) + 1)
                    append!(list_m2_v, 1)
                end

                if ii > 0
                    append!(list_m2_x, jj * param.N_x + ii + 1)
                    append!(list_m2_y, (jj) * param.N_x + (ii - 1) + 1)
                end

                if jj + 1 < param.N_y && ii + 1 < param.N_x
                    append!(list_m3_x, jj * param.N_x + ii + 1)
                    append!(list_m3_y, (jj + 1) * param.N_x + (ii + 1) + 1)
                end

                if jj > 0 && ii > 0
                    append!(list_m3_x, jj * param.N_x + ii + 1)
                    append!(list_m3_y, (jj - 1) * param.N_x + (ii - 1) + 1)
                end

                if jj + 1 < param.N_y
                    append!(list_m4_x, jj * param.N_x + ii + 1)
                    append!(list_m4_y, (jj + 1) * param.N_x + ii + 1)
                end

                if jj > 0
                    append!(list_m4_x, jj * param.N_x + ii + 1)
                    append!(list_m4_y, (jj - 1) * param.N_x + ii + 1)
                end

                if jj + 1 < param.N_y && ii > 0
                    append!(list_m5_x, jj * param.N_x + ii + 1)
                    append!(list_m5_y, (jj + 1) * param.N_x + (ii - 1) + 1)
                end

                if jj > 0 && ii + 1 < param.N_x
                    append!(list_m5_x, jj * param.N_x + ii + 1)
                    append!(list_m5_y, (jj - 1) * param.N_x + (ii + 1) + 1)
                end
            end
        end
    end

    m2_m = SparseTensor(list_m2_x, list_m2_y, ones(size(list_m2_y)), param.Ne, param.Ne)
    m3_m = SparseTensor(list_m3_x, list_m3_y, ones(size(list_m3_y)), param.Ne, param.Ne)
    m4_m = SparseTensor(list_m4_x, list_m4_y, ones(size(list_m4_y)), param.Ne, param.Ne)
    m5_m = SparseTensor(list_m5_x, list_m5_y, ones(size(list_m5_y)), param.Ne, param.Ne)

    m2_m_1 = SparseTensor(list_m2_x, list_m2_x, ones(size(list_m2_x)), param.Ne, param.Ne)
    m3_m_1 = SparseTensor(list_m3_x, list_m3_x, ones(size(list_m3_x)), param.Ne, param.Ne)
    m4_m_1 = SparseTensor(list_m4_x, list_m4_x, ones(size(list_m4_x)), param.Ne, param.Ne)
    m5_m_1 = SparseTensor(list_m5_x, list_m5_x, ones(size(list_m5_x)), param.Ne, param.Ne)

    #Building matrices for applying BC
    ii_l = [i for i in 1:param.N_x:param.Ne]
    jj_l = [1 for i in 1:param.N_x:param.Ne]
    vv_l = [1.0 for i in 1:param.N_x:param.Ne]
    # B
    A_l_m = SparseTensor(ii_l, jj_l, vv_l, param.Ne, 1)

    ii_r = [i for i in param.N_x:param.N_x:param.Ne]
    jj_r = [1 for i in 1:param.N_x:param.Ne]
    vv_r = [1.0 for i in 1:param.N_x:param.Ne]
    A_r_m = SparseTensor(ii_r, jj_r, vv_r, param.Ne, 1)
    A_m = 1 - Array(A_l_m) - Array(A_r_m)

    aux_matrix(ii_l, jj_l, vv_l, ii_r, jj_r, vv_r, m2_m, m3_m, m4_m, m5_m, m2_m_1, m3_m_1, m4_m_1, m5_m_1, A_m)
end

function load_K_s(model_param)
    # function to load the initial guess of Ks
    files_x = string("./", model_param.run_name, "/K_x", ".txt")
    files_y = string("./", model_param.run_name, "/K_y", ".txt")

    k_x_list = readdlm(files_x, ' ', Float64)
    k_y_list = readdlm(files_y, ' ', Float64)

    n_step_skip = model_param.N_steps_orig ÷ model_param.N_steps

    k_x_list = k_x_list[1:n_step_skip:1800] * model_param.K_mupltiplier
    k_y_list = k_y_list[1:n_step_skip:1800] * model_param.K_mupltiplier

    return k_x_list, k_y_list
end

function load_QoIs(model_param)
    # function to load the quantity of interest and calculate the standard deviation

    files_x = [string("./", model_param.run_name, "/Q_x_mid", ".txt")]
    files_y = [string("./", model_param.run_name, "/Q_y_mid", ".txt")]
    # files_y = [string("./",model_param.run_name,"/Q_y_",string(ii),".txt") for ii =1:9]

    y_x_list = [readdlm(files_x[ii], ' ', Float64) for ii = 1]
    y_y_list = [readdlm(files_y[ii], ' ', Float64) for ii = 1]
    # y_y_list = [readdlm(files_y[ii], ' ', Float64) for ii =1:9];

    momment2 = maximum([maximum(std(y_x_list))])
    n_step_skip = model_param.N_steps_orig ÷ model_param.N_steps

    y_x_list = [y_x_list[ii][:, 1:n_step_skip:1800] for ii = 1]# automate the time steps
    y_y_list = [y_y_list[ii][:, 1:n_step_skip:1800] for ii = 1]# automate the time steps

    # y_y_list = [y_y_list[ii][:,1:n_step_skip:1800] for ii =1:9];

    return momment2, y_x_list, y_y_list
end
end


