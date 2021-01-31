using DataStructures
using Dates
using Flux
using Flux.Optimise: update!
using OpenAIGym
using Printf
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

const newline_frequency = 60

function run()
    q = Q()
    explore_policy = Policy(0.5, q)
    exploit_policy = Policy(0.0, q)
    explore_rewards = Float32[]
    exploit_rewards = Float32[]
    time_steps = 0
    start_time = now()
    newline_time = time() + newline_frequency
    try
        policy = Policy(0.5, Q())
        sars = CircularBuffer{SARSF{Vector{Float32},Int8}}(10_000)
        for episode in 1:2000
            episode_t = 0
            explore_r = run_episode(env, explore_policy) do (s, a, r, s′)
                time_steps += 1
                episode_t += 1
                @assert episode_t <= env_step_limit
                failed = finished(env) && episode_t < env_step_limit
                push!(sars, SARSF(
                    Float32.(copy(s)),
                    Int8(a),
                    Float32(r),
                    Float32.(copy(s′)),
                    failed))
                optimize!(policy, sample(sars, 64))
                #render(env)
            end
            episode_t = 0
            exploit_r = run_episode(env, exploit_policy) do _
                episode_t += 1
                @assert episode_t < env_step_limit
            end
            push!(explore_rewards, explore_r)
            push!(exploit_rewards, exploit_r)
            @printf("\u1b[?25l\u1b[0E%s ep %5d ts %6d, explore r %6.2f ± %6.2f, exploit r %6.2f ± %6.2f \u1b[0K\u1b[?25h",
                    Dates.format(Time(Nanosecond(now() - start_time)), "HH:MM:SS"),
                    episode, time_steps,
                    mean(last(explore_rewards, 100)), std(last(explore_rewards, 100)),
                    mean(last(exploit_rewards, 100)), std(last(exploit_rewards, 100)))
            if time() >= newline_time
                newline_time += newline_frequency
                println()
            end
            if mean(last(exploit_rewards, 100)) >= 475; break end
        end
    catch e
        if !isa(e, InterruptException); rethrow() end
    finally
        close(env)
    end
    (exploit_policy, exploit_rewards, explore_rewards)
end

last(xs, n) = xs[max(1, end-n+1):end]

function demo_policy(policy, n_episodes=5)
    try
        return map(1:n_episodes) do _
            run_episode(env, policy) do _
                render(env)
            end
        end
    catch e
        if !isa(e, InterruptException); rethrow() end
    finally
        close(env)
    end
end

function graph(rewards)
    scatter(rewards, size=(1200, 800), background_color=:black, markercolor=:white, legend=false,
            markersize=3, markeralpha=0.3,
            markerstrokewidth=0, markerstrokealpha=0)
end
