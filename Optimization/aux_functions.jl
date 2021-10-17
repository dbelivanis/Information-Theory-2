# Definition of functions for the forward and backward problem

module aux_functions
include("./param.jl")
export Darcy_flow_solver, Info_upscale,print_status,save_values,update_K_p

using Main.param

using ADCME
using SparseArrays  

using DelimitedFiles
using Dates
using Plots

    function expit(x)
        # Definition of expit funtion
        1 / (1+ exp(-x))
    end

    function left_BC(t)
        # Definition of the left boundary condition of time_step t

        t_norm = (t/model_param.T * 10 -0.5)*180
        time_multiplier = exp(-t_norm/50)/30
    end
    
    function BC(t)
        # Definition of the boundary condition of time_step t

        BC_right_ = zeros(model_param.N_y)
        H = map(expit,-LinRange(-31.5,31.5,model_param.N_y)/10) /2 .+ 0.2

        BC_left_ = H .* left_BC(t) + LinRange(0.2,0,model_param.N_y) .+ 1
        
        BC_left = SparseTensor(aux_matrix.ii_l, aux_matrix.jj_l, BC_left_, model_param.Ne, 1) 
        BC_right = SparseTensor(aux_matrix.ii_r, aux_matrix.jj_r, BC_right_, model_param.Ne, 1)
                
        return BC_left, BC_right, BC_left_, BC_right_  
    end

    function IC()
        # Definition of the Initial Condition
        
        _,_ , BC_left, Bc_right = BC(0)
        h_0 = reshape((reshape(BC_left,(model_param.N_y,1)) .- reshape(BC_left,(model_param.N_y,1)) * reshape(constant(x),(1,model_param.N_x))),(1,model_param.Ne)).* ones(model_param.N_k,1)
    
    end
   
    function advance_time(a,b,c,h_)
        # Definition of function to advance time

        h_n = TensorArray(model_param.N_k)
        j = constant(1,dtype=Int32)
        
        
        aa = tf.slice(a,[j-1,0],[1,1])[1,1]
        bb = tf.slice(b,[j-1,0],[1,1])[1,1]
        cc = tf.slice(c,[j-1,0],[1,1])[1,1]
    
        dd = (2 * aa * bb) / (aa+bb) 
    
        m2 = -aa + cc^2 / dd
        m3 = -cc/2 - cc^2 / (2 * dd)
        m4 = -bb + cc^2 / dd
        m5 = -cc/2 - cc^2 / (2 * dd)
    
        A = m2 * aux_matrix.m2_m + m3 * aux_matrix.m3_m + m4 * aux_matrix.m4_m + 
        m5 * aux_matrix.m5_m + m2 * aux_matrix.m2_m_1 + m3 * aux_matrix.m3_m_1 +
        m4 * aux_matrix.m4_m_1 + m5 * aux_matrix.m5_m_1 + spdiag(model_param.Ne)
    
    
        function condition(j,h_n)
            # run for all the tensor that are less than N_k_dis
            j <= N_k_dis
        end
    
        function body(j,h_n)   
            # function to solve and advance time for each part of tensor
            # build lhs
            aa = tf.slice(a,[j-1,0],[1,1])[1,1]
            bb = tf.slice(b,[j-1,0],[1,1])[1,1]
            cc = tf.slice(c,[j-1,0],[1,1])[1,1]
            
            dd = (2 * aa * bb) / (aa+bb) 
            
            m2 = -aa + cc^2 / dd
            m3 = -cc/2 - cc^2 / (2 * dd)
            m4 = -bb + cc^2 / dd
            m5 = -cc/2 - cc^2 / (2 * dd)
            
        
            A = m2 * aux_matrix.m2_m + m3 * aux_matrix.m3_m + m4 * aux_matrix.m4_m + 
            m5 * aux_matrix.m5_m - (m2 * aux_matrix.m2_m_1 + m3 * aux_matrix.m3_m_1 +
            m4 * aux_matrix.m4_m_1 + m5 * aux_matrix.m5_m_1 - spdiag(model_param.Ne))      
            
            # solve system
            
            h_j = A\h_[j]
            
            # update head
            h_n = write(h_n,j,h_j)
    
            j+1, h_n
        end
        
        # Definition of while loop of tensorflow
        _, out = while_loop(condition, body, [j, h_n])
        
        return stack(out)
    
    end
    
    ## aux functions for flux calculations
    function P(h_,i,j)
        # get the head for specifi i and j
        tf.slice(h_,constant([0,(j-1)*model_param.N_x+i-1], dtype=Int32),[-1,1])
    end
        
    function get_hor_flux(a,b,c,h_,i,j)
        # calculate horizontal flux for given i and j
        ((a    .- tf.square(c) ./(2*b)) .* (P(h_,i,j)-P(h_,i+1,j)) +
        (c./4 .+ tf.square(c) ./(4*b)) .* (P(h_,i-1,j-1)-P(h_,i+1,j+1)) +
        (c./4 .- tf.square(c) ./(4*b)) .* (P(h_,i-1,j+1)-P(h_,i+1,j-1)) )/model_param.dx
    end
    
    function get_ver_flux(a,b,c,h_,i,j)
        # calculate vertikal flux for given i and j
        
        ((b    .- tf.square(c) ./(2*a)) .* (P(h_,i,j)-P(h_,i,j+1)) +
        (c./4 .+ tf.square(c) ./(4*a)) .* (P(h_,i+1,j-1)-P(h_,i-1,j+1)) +
        (c./4 .- tf.square(c) ./(4*a)) .* (P(h_,i-1,j-1)-P(h_,i+1,j+1)) )/model_param.dy
    
    end

    mutable struct tf_variables_definition
        # structrure with all the variables are optimized and placeholder
        lambda
        N_k_dis
        k_x_t
        k_y_t
        k_xy_t
        k_x_t_log
        k_y_t_log
        k_xy_t_log

    end

    ## Main model constructor
    function Darcy_flow_solver(model_param_local)
        # Main function to solve the Darcy flow problem - forward problem

        # Definition of basic parameters
        global model_param = model_param_local
        global aux_matrix = Multi_point_flux_aux_matrix(model_param)
        global x = LinRange(0,1,model_param.N_x)
        global y = LinRange(0,1,model_param.N_y)
        global N_k_dis = placeholder(model_param.N_k,dtype=Int32)
        lambda = placeholder(ones(1))
       
        # initialize the head with initial condition
        h_IC = IC()
        # Initialize the tensor of head for all time steps
        h_t = TensorArray(model_param.N_steps)
        # load values of initial guess of the permeability
        k_x, k_y = param.load_K_s(model_param)
        
        # Initialize the variables of permeabilities
        k_x_t_log = Variable(log.(reshape(k_x,(1,model_param.N_steps))  .+ zeros(model_param.N_k,model_param.N_steps)).+ 4e-3.*(0.5 .- rand(model_param.N_k,model_param.N_steps)))
        k_x_t = tf.exp(k_x_t_log)

        k_y_t_log = Variable(log.(reshape(k_y,(1,model_param.N_steps))  .+ zeros(model_param.N_k,model_param.N_steps)).+ 4e-3.*(0.5 .- rand(model_param.N_k,model_param.N_steps)))
        k_y_t = tf.exp(k_y_t_log)

        k_xy_t_log = Variable(zeros(model_param.N_k,model_param.N_steps)  .+ 1e-5.*(0.5 .- rand(model_param.N_k,model_param.N_steps)))
        k_xy_t = tf.tanh(k_xy_t_log) .* k_x_t^0.5 .* k_y_t^0.5 .* 0.4


        # Initialize the tensor o flow in x and y direction
        q_t_x = [TensorArray(model_param.N_steps) for ii = 1:model_param.N_points]
        q_t_y = [TensorArray(model_param.N_steps) for ii = 1:model_param.N_points];



        # Initial state out of the Loop 
        # Initial condition
        
        # write the first head value in the tensor
        h_t = write(h_t, 1,constant(h_IC))

        # Extract the permeabilities
        k_x = tf.slice(k_x_t,constant([0,0],dtype=Int32),[-1,1]) 
        k_y = tf.slice(k_y_t,constant([0,0],dtype=Int32),[-1,1]) 
        k_xy = tf.slice(k_xy_t,constant([0,0],dtype=Int32),[-1,1]) 

        # Evaluate the flows based on initial condition
        q_y = [get_ver_flux(k_x,k_y,k_xy,h_IC,model_param.loc_x_list[ii],model_param.loc_y_list[ii]) for ii in 1:length(model_param.loc_x_list)]
        q_x = [get_hor_flux(k_x,k_y,k_xy,h_IC,model_param.loc_x_list[ii],model_param.loc_y_list[ii]) for ii in 1:length(model_param.loc_x_list)]

        # Write the initial flows on tensor
        q_t_x = [write(q_t_x[ii],1,q_x[ii]) for ii = 1:model_param.N_points];
        q_t_y = [write(q_t_y[ii],1,q_y[ii]) for ii = 1:model_param.N_points];

        # next step
        i = constant(2, dtype=Int32)

        
        S = 1e-6 # Storativity

            function condition(i, h_t_loop, q_t_x_loop, q_t_y_loop)
                # for loop until last time step
                i<= model_param.N_steps
            end

            function body(i, h_t_loop, q_t_x_loop, q_t_y_loop)
                
                # calculate time
                t = cast(i-1,Float64)*model_param.dt

                # build rhs
                h_rhs = read(h_t_loop, i-1)
                BC_left, BC_right = BC(t)
                h_rhs = h_rhs .* aux_matrix.A_m[:,1] + Array(BC_left)[:,1] + Array(BC_right)[:,1]

                #extract permeability
                k_x = tf.slice(k_x_t,[0,i-1],[-1,1]) 
                k_y = tf.slice(k_y_t,[0,i-1],[-1,1])
                k_xy = tf.slice(k_xy_t,[0,i-1],[-1,1])

                # advance time
                h_next = advance_time(k_x/S,k_y/S,k_xy/S,h_rhs)

                # # # update hydraulic head
                h_t_loop = write(h_t_loop, i, h_next)

                # Evaluate the flows on the corresponding time step
                q_x = [get_hor_flux(k_x,k_y,k_xy,h_next,model_param.loc_x_list[ii],model_param.loc_y_list[ii]) for ii in 1:length(model_param.loc_x_list)]
                q_y = [get_ver_flux(k_x,k_y,k_xy,h_next,model_param.loc_x_list[ii],model_param.loc_y_list[ii]) for ii in 1:length(model_param.loc_x_list)]
                
                # Write the flows of the corresponding time step to the tensor
                q_t_x_loop = [write(q_t_x_loop[ii],i,q_x[ii]) for ii = 1:model_param.N_points];
                q_t_y_loop = [write(q_t_y_loop[ii],i,q_y[ii]) for ii = 1:model_param.N_points];
                
                # return head and flow
                i+1, h_t_loop, q_t_x_loop, q_t_y_loop
                
            end

        # Define the tensor flow loop
        _, out, outx, outy = while_loop(condition, body, [i, h_t,q_t_x, q_t_y])

        h_t = stack(out)
        q_t_x = [stack(outx[i]) for i=1:model_param.N_points]
        q_t_y = [stack(outy[i]) for i=1:model_param.N_points];

        # Pack the important parameters for later use
        tf_variables = tf_variables_definition(lambda,N_k_dis,k_x_t,k_y_t,k_xy_t,k_x_t_log,k_y_t_log,k_xy_t_log)
        return tf_variables, h_t, q_t_x, q_t_y

    end

    ## Probability constructions
    function Info_upscale(tf_variables,model_param,q_t_x,q_t_y,N_k_dis_,maxiter=400)
        # function to construct the loss funtion for the optimization problem

        # Define probabilities as parameters of softmax to assure sum(p)=1
        p_pre_soft_max_values = ones(1,model_param.N_k);# .+ (1e0 .* rand(1,model_param.N_k)) ; #CHANGED TO CHECK IF NOT OPTIMIZED --

        # Load quantity of interest that are needed for the optimization process
        momment2, y_x_list, y_y_list = load_QoIs(model_param)

        # Make only part of the probabilities to be active
        for i = 1:N_k_dis_
            p_pre_soft_max_values[1,i] +=100
        end

        # Pass values in to soft max to make them as probabilities
        p_pre_soft_max = Variable(p_pre_soft_max_values, trainable=false) #CHANGED TO CHECK IF NOT OPTIMIZED
        p = tf.nn.softmax(p_pre_soft_max,1)

        # Evaluation of all loss functions for all available points
        loss_x_list = [loss_function(tf_variables.lambda,p,y_x_list[ii],q_t_x[ii]) for ii = 1:model_param.N_points]
        loss_y_list = [loss_function(tf_variables.lambda/10,p,y_y_list[ii],q_t_y[ii]) for ii = 1:model_param.N_points];

        # Evaluate the loss function for the central point
        loss_x = loss_x_list[5][1] + loss_x_list[4][1] + loss_x_list[6][1] #+ loss_x_list[2][1] + loss_x_list[8][1] 
        loss_y = loss_y_list[5][1] + loss_y_list[4][1] + loss_y_list[6][1] #+ loss_y_list[2][1] + loss_y_list[8][1] 
        loss = loss_x + loss_y 
        
        # Loss function as Mean Square Error
        dw_x = loss_x_list[5][2] + loss_x_list[4][2] + loss_x_list[6][2] #+ loss_x_list[2][2] + loss_x_list[8][2] 
        dw_y = loss_y_list[5][2] + loss_x_list[4][2] + loss_y_list[6][2] #+ loss_x_list[2][2] + loss_y_list[8][2]         
        dw_2_sum = dw_x + dw_y #change name


        # Evaluation of the distance of consecutive flow values 
        sort_list = [tf.sort(tf.slice(q_t_x[ii],[0,0,0],[model_param.N_steps,tf_variables.N_k_dis,1]),axis=1) for ii=5]
        diff_list = [tf.reduce_max(tf.reduce_min((tf.slice(sort_list[ii],[0,1,0],[-1,-1,-1])-
        (tf.slice(sort_list[ii],[0,0,0],[-1,tf_variables.N_k_dis-1,-1])))./tf.reduce_mean(sort_list[ii],axis=1,keep_dims=true),axis=1)) for ii =1]
        diff_eval =  tf.reduce_max(tf.stack(diff_list))

        # Define all the optimization algorithm ADAM and LFGS for both MSE and information theory approach
        opt_ADAM = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss* 1e5) 
        opt_LFGS = ScipyOptimizerInterface(loss* 1e5; method="L-BFGS-B", options=Dict("maxiter"=> maxiter * 2, "ftol"=>1e-12, "gtol"=>1e-12))
        opt_ADAM_sum = tf.train.AdamOptimizer(learning_rate=0.001).minimize(dw_2_sum)
        opt_LFGS_sum = ScipyOptimizerInterface(dw_2_sum * 1e5; method="L-BFGS-B", options=Dict("maxiter"=> maxiter, "ftol"=>1e-12, "gtol"=>1e-12))

        return loss,dw_2_sum , opt_ADAM, opt_LFGS, opt_ADAM_sum, opt_LFGS_sum, diff_eval,p_pre_soft_max, p
    end

    function loss_function(lambda,p,y_t,q_t)
        # Definition of loss function for information theory upscale
    
        q_t = tf.slice(q_t,constant([0,0,0],dtype=Int32),constant([-1,-1,1],dtype=Int32))
    
        dw = tf.squared_difference(constant(reshape(transpose(y_t),(model_param.N_steps,model_param.N_k_fine,1))),constant(reshape(q_t,(model_param.N_steps,1,model_param.N_k))))
    
        # dw_2_sum = tf.reduce_mean(dw) * 1e5 # devide with momment
    
        dw_2_mean = tf.reduce_sum(tf.reduce_mean(dw,axis=0).*p)/model_param.N_k_fine

        momment2 = reshape(std(y_t,dims=1),(model_param.N_steps,1,1)).^2;
    
        dw = tf.reduce_sum(dw ./ momment2,axis=0)
        dw_2_sum = tf.reduce_mean(dw) * 1e5
        # dw = tf.reduce_sum(dw ,axis=0)
    
        dw_min = tf.reduce_min(dw,1,keepdims=true)
        dw_m_min = -dw + dw_min
    
        F = p .* tf.exp(dw_m_min/lambda)
    
        F = tf.reduce_sum(F,1,keep_dims=true)
    
        F = tf.log(F)- dw_min/lambda
    
        loss = - tf.reduce_sum(F)/model_param.N_k_fine

        return loss , dw_2_sum
    end

    function update_K_p(sess,model_param,tf_variables,check_diff,N_k_dis_,p_pre_soft_max)
        # function to update the permeabilities, if criteria are met the active probabilities are doubled otherwise the permeabilities are perturbed 
        k_x_t_log = tf_variables.k_x_t_log
        k_y_t_log = tf_variables.k_y_t_log
        k_xy_t_log = tf_variables.k_xy_t_log
        N_k = model_param.N_k

        print("function for update K:",check_diff,"\t",N_k_dis_,"\n")
        if check_diff > 5e-5 && N_k_dis_ < N_k
                 
            k_x_t_update = run(sess,k_x_t_log) 
            k_x_t_update[N_k_dis_+1:N_k_dis_*2,:] = k_x_t_update[1:N_k_dis_,:] 
            k_x_t_update = k_x_t_update .+ (0.0 .+ 5e-3 .* (0.5 .- rand(N_k,model_param.N_steps
        )))
                    run(sess,tf.assign(k_x_t_log,k_x_t_update));
                    
                    k_y_t_update = run(sess,k_y_t_log) 
                    k_y_t_update[N_k_dis_+1:N_k_dis_*2,:] = k_y_t_update[1:N_k_dis_,:] 
                    k_y_t_update = k_y_t_update .+ (0.0 .+ 5e-5.* (0.5 .- rand(N_k,model_param.N_steps
        )))
                    run(sess,tf.assign(k_y_t_log,k_y_t_update));   
                        
                    k_xy_t_update = run(sess,k_xy_t_log) 
                    k_xy_t_update[N_k_dis_+1:N_k_dis_*2,:] = k_xy_t_update[1:N_k_dis_,:] 
                    k_xy_t_update = k_xy_t_update .+ (0.0 .+ 0e-5 .* (0.5 .- rand(N_k,model_param.N_steps
        )))
                    run(sess,tf.assign(k_xy_t_log,k_xy_t_update));
                
                    p_pre_soft_max_update = run(sess,p_pre_soft_max) 
                    p_pre_soft_max_update[:,N_k_dis_+1:N_k_dis_*2] = p_pre_soft_max_update[:,1:N_k_dis_] 
                    p_pre_soft_max_update = p_pre_soft_max_update .* (1.0 .+ 0 *  (rand(1,N_k).-0.5))
                    run(sess,tf.assign(p_pre_soft_max,p_pre_soft_max_update));
                    
                    N_k_dis_ *=2
                else
                    k_x_t_update = run(sess,k_x_t_log) .+ (0.0 .+5e-5 *  (rand(N_k,model_param.N_steps
        ).-0.5));
                    run(sess,tf.assign(k_x_t_log,k_x_t_update));
                        
                    k_y_t_update = run(sess,k_y_t_log) .+ (0.0 .+5e-5 *  (rand(N_k,model_param.N_steps
        ).-0.5));
                    run(sess,tf.assign(k_y_t_log,k_y_t_update));
                        
                    k_xy_t_update = run(sess,k_xy_t_log) .+ (0.0 .+0e-5 *  (rand(N_k,model_param.N_steps
        ).-0.5));
                    # run(sess,tf.assign(k_xy_t_log,k_xy_t_update));
                        
                    p_pre_soft_max_update = run(sess,p_pre_soft_max) .* (1 .+ 0 *  (rand(1,N_k).-0.5));
                    run(sess,tf.assign(p_pre_soft_max,p_pre_soft_max_update));
                end
                    
                return N_k_dis_
    end

    function multiply_K(sess,model_param,tf_variables,multiplier)
        # function to shift permeability, for debuging purposes
        k_x_t_log = tf_variables.k_x_t_log
        k_y_t_log = tf_variables.k_y_t_log
        k_xy_t_log = tf_variables.k_xy_t_log
        N_k = model_param.N_k
               
        k_x_t_update = run(sess,k_x_t_log) 
        k_x_t_update = k_x_t_update .+ multiplier
        run(sess,tf.assign(k_x_t_log,k_x_t_update));
                
        k_y_t_update = run(sess,k_y_t_log) 
        k_y_t_update = k_y_t_update .+ multiplier
        # run(sess,tf.assign(k_y_t_log,k_y_t_update));   
                    
        k_xy_t_update = run(sess,k_xy_t_log) 
        k_xy_t_update = k_xy_t_update .* multiplier
        # run(sess,tf.assign(k_x_t_log,k_x_t_update));
        
    end

    function initialize_sess()
        #function to initialize tensor flow session
        sess = Session(); init(sess)
    end
    
    function optimize(T_,N_k_dis_)
        # fucntion to run optimization
        ScipyOptimizerMinimize(sess, opt_LFGS_sum,feed_dict = Dict(lambda => ones(1)*T_,N_k_dis=>N_k_dis_))
    end

    function save_values(sess,model_param,tf_variables,q_t_x, q_t_y,p,T_exp,mode="a")
        # function to save values of permeabilities during optimization
        
        # name of current experiment
        exp_name = model_param.exp_name
        # evaluation of tensor flow variables and tensors
        k_x_save = run(sess,tf_variables.k_x_t)
        k_y_save = run(sess,tf_variables.k_y_t)
        k_xy_save = run(sess,tf_variables.k_xy_t)
        p_save = run(sess,p)
        q_x_save = run(sess,q_t_x[5])
        q_y_save = run(sess,q_t_y[5])

        # Write in each txt the corresponding value
        open(string("./../results/",exp_name,"lambda.txt"),mode) do io
                writedlm(io, T_exp)
        end

        open(string("./../results/",exp_name,"k_x.txt"),mode) do io
                writedlm(io, k_x_save)
        end
        open(string("./../results/",exp_name,"k_y.txt"),mode) do io
                writedlm(io, k_y_save)
        end
        open(string("./../results/",exp_name,"k_xy.txt"),mode) do io
            writedlm(io, k_xy_save)
        end
        open(string("./../results/",exp_name,"p.txt"), mode) do io
                writedlm(io, p_save)
        end
        open(string("./../results/",exp_name,"q_x.txt"),mode) do io
                writedlm(io, q_x_save)
        end
        open(string("./../results/",exp_name,"q_y.txt"), mode) do io
                writedlm(io, q_y_save)
        end
    end

    function print_status(sess,model_param,loss,diff_eval,T_exp,T_,N_k_dis_,tf_variables,mode="a")
        # Funtion to print the update for debuging and experiment evolution check
        
        exp_name = model_param.exp_name
        diff_,loss_ = run(sess,[diff_eval,loss],feed_dict = Dict(tf_variables.lambda => ones(1)*T_,tf_variables.N_k_dis=>N_k_dis_))
        print("Saving Values at: ",Dates.format(now(), "HH:MM") ,T_exp,"\t",T_,"\t",diff_,"\t",loss_,"\t","\t",N_k_dis_,"\n")  

        open(string("./../results/",exp_name,"update.txt"), mode) do io
               print("Saving Values at: ",Dates.format(now(), "HH:MM") ,T_exp,"\t",T_,"\t",diff_,"\t",loss_,"\t","\t",N_k_dis_,"\n")  
        end
    end
end