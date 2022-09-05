#=
test:
- Julia version: 1.8.0
- Author: andy
- Date: 2022-09-02
=#
using LevyArea
using DelimitedFiles

function gen_samples(;its::Int64 = 65536,m::Int64 = 2,h:: Float64 = 0.001,err:: Float64 = 0.005,fixed:: Bool = false, W:: Array{Float64} = [1.0,-0.5,-1.2,-0.3,0.7,0.2,-0.9,0.1,1.7])

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

        if i%100 == 0
            println(i)
        end
    end

    filename = "samples/samples_$m-dim.csv"
    if fixed
        filename = "samples/fixed_samples_$m-dim.csv"
    end

    writedlm(filename, results, ',')
end
