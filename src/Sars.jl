module Sars

export SARSF, stack, onehot

struct SARSF{S, A}
    s  :: S
    a  :: A
    r  :: Float32
    s′ :: S
    f  :: Bool
end

struct SARSFStack
    s     :: Array{Float32, 2}
    a_hot :: Array{Float32, 2}
    r     :: Array{Float32, 2}
    s′    :: Array{Float32, 2}
    f     :: Array{Float32, 2}
end

function stack(sars::Vector{SARSF{S, A}}, n_actions, a_to_i) where {S, A}
    SARSFStack(
        reduce(hcat, map(x -> x.s, sars)),
        reduce(hcat, map(x -> onehot(a_to_i(x.a), n_actions), sars)),
        reduce(hcat, map(x -> x.r, sars)),
        reduce(hcat, map(x -> x.s′, sars)),
        reduce(hcat, map(x -> x.f ? 0.0f0 : 1.0f0, sars))
    )
end

function onehot(i, length)
    result = zeros(Float32, length)
    result[i] = 1.0f0
    result
end

end # end module
