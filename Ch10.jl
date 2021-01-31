using DataStructures
using Flux
using Flux.Optimise: update!
using OpenAIGym
using Statistics

const env = GymEnv(:CartPole, :v1)

struct Q{N, A, V}
    main_network   :: N
    action_network :: A
    value_network  :: V
end
Flux.@functor Q
function Q()
    main_network = Chain(
        Dense(length(env.state), 512, relu,
              initW=Flux.kaiming_uniform, initb=Flux.kaiming_uniform),
        Dense(512, 128, relu,
              initW=Flux.kaiming_uniform, initb=Flux.kaiming_uniform))
    action_network = Dense(128, length(env.actions), relu,
                           initW=Flux.kaiming_uniform, initb=Flux.kaiming_uniform)
    value_network = Dense(128, 1, identity,
                          initW=Flux.kaiming_uniform, initb=Flux.kaiming_uniform)
    Q(main_network, action_network, value_network)
end
function (Q::Q)(s)
    n = Q.main_network(s)
    a = Q.action_network(n)
    v = fill(Q.value_network(n)[1], size(a))
    v + a .- mean(a)
end

struct Policy{Q} <: AbstractPolicy
    ϵ        :: Float32
    q_online :: Q
    q_target :: Q
end
Policy(ϵ, q) = Policy(Float32(ϵ), q, deepcopy(q))

env_a_to_network_a(a) = a+1
network_a_to_env_a(a) = a-1

function Reinforce.action(policy::Policy, r, s, A)
    if rand() < policy.ϵ
        rand(A)
    else
        argmax(policy.q_online(s)) |> network_a_to_env_a
    end
end

struct SARSF{S, A}
    s      :: S
    a      :: A
    r      :: Float32
    s′     :: S
    failed :: Bool
end

const env_step_limit = env.pyenv._max_episode_steps

function onehot(hot_i, length)
    result = zeros(Float32, length)
    result[hot_i] = 1.0f0
    result
end

const opt = RMSProp(0.000_7)

function polyak_average!(a, b, τ=0.01)
    for (pa, pb) in zip(a, b)
        pa *= 1 - τ
        pa += pb * τ
    end
end

function optimize!(policy, sars, γ=1.0f0)
    γ = Float32(γ)
    s = reduce(hcat, map(x -> x.s, sars))
    a = reduce(hcat, 
               map(x -> onehot(env_a_to_network_a(x.a), length(env.actions)), sars))
    r = reshape(map(x -> x.r, sars), 1, :)
    s′ = reduce(hcat, map(x -> x.s′, sars))
    f = reshape(map(x -> x.failed ? 0.0f0 : 1.0f0, sars), 1, :)
    @assert typeof(s) == Array{Float32, 2}
    @assert typeof(a) == Array{Float32, 2}
    @assert typeof(r) == Array{Float32, 2}
    @assert typeof(s′) == Array{Float32, 2}
    @assert typeof(f) == Array{Float32, 2}
    @assert typeof(γ) == Float32
    a′ = mapslices(
        eavs -> onehot(argmax(eavs), length(eavs)),
        policy.q_online(s′),
        dims=1)
    @assert typeof(a′) == Array{Float32, 2}
    qs′ = policy.q_target(s′)
    @assert typeof(qs′) == Array{Float32, 2}
    target = r + γ * sum(qs′ .* a′, dims=1) .* f
    grads = gradient(params(policy.q_online)) do
        qs = policy.q_online(s)
        predicted = sum(qs .* a, dims=1)
        mean((target - predicted).^2)
    end
    update!(opt, params(policy.q_online), grads)
    polyak_average!(params(policy.q_target), params(policy.q_online))
end

function run()
    try
        policy = Policy(0.5, Q())
        global sars = CircularBuffer{SARSF{Vector{Float32},Int8}}(10_000)
        for _ in 1:50
            episode_t = 0
            r = run_episode(env, Policy(0.5f0, Q())) do (s, a, r, s′)
                episode_t += 1
                @assert episode_t <= env_step_limit
                failed = finished(env) && episode_t < env_step_limit
                push!(sars, SARSF(
                    Float32.(copy(s)),
                    Int8(a),
                    Float32(r),
                    Float32.(copy(s′)),
                    failed))
                optimize!(policy, sars)
                render(env)
            end
        end
    catch e
        if !isa(e, InterruptException); rethrow() end
    finally
        close(env)
    end
end
