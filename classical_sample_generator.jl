#=
test:
- Julia version: 1.8.0
- Author: andy
- Date: 2022-09-02
=#
using LevyArea
using DelimitedFiles

function gen_samples(;its::Int64 = 65536,m::Int64 = 2,h:: Float64 = 1.0,
    err:: Float64 = 0.0001,fixed:: Bool = false,
    W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7],
    filename = "")

    resDim = Int64(m*(m+1)/2)
    results = Array{Float64}(undef,its,resDim)
    W = W[1:m]


    for i in 1:its

        if fixed == false
           W = randn(m)
        end
        II = iterated_integrals(W,h,err)

        idx:: Int = m+1
        for k in 1:m
            results[i,k] = W[k]
            #println("wiritng results[$i,$k] = $(W[k])")
            for l in (k+1):m
                a = 0.5*(II[k,l] - II[l,k])
                results[i,idx] = a
                idx +=1
            end
        end

        # if i%100 == 0
        #     println(i)
        # end
    end

    if filename == ""
        filename = "samples/samples_$m-dim.csv"
        if fixed
            filename = "samples/fixed_samples_$m-dim.csv"
        end
    end

    # filename = "high_prec_samples.csv"

    writedlm(filename, results, ',')
end

function time_measure(its, m, h, err)
    W = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7]
    W = W[1:m]
    for i in 1:its
        iterated_integrals(W,h,err)
    end
end

function generate_all()
    for i in 2:8
        gen_samples(its = 1048576, m = i)
        gen_samples(m = i, fixed = true)
        println("$i done")
    end
end

function list_pairs(m)
    resDim = Int64(m*(m-1)/2)
    res = Array{Tuple{Int64,Int64}}(undef,resDim)
    idx:: Int = 1
    for k in 1:m
        for l in (k+1):m
            res[idx] = (k,l)
            idx +=1
        end
    end
    println(res)
end
