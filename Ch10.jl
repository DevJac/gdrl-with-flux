using DataStructures
using Dates
using Flux
using Flux.Optimise: update!
using OpenAIGym
using Plots
using Printf
using Statistics
using Sars

const env = GymEnv(:CartPole, :v1)

function Linear(in, out, activation)
    Dense(in, out, activation,
          initW=(_dims...) -> Float32.((rand(out, in).-0.5).*(2/sqrt(in))),
          initb=(_dims...) -> Float32.((rand(out).-0.5).*(2/sqrt(in))))
end

struct Q{N, A, V}
    main_network   :: N
    action_network :: A
    value_network  :: V
end
Flux.@functor Q
function Q()
    main_network = Chain(
        Linear(length(env.state), 512, relu),
        Linear(512, 128, relu))
    action_network = Linear(128, length(env.actions), identity)
    value_network = Linear(128, 1, identity)
    Q(main_network, action_network, value_network)
end
function (Q::Q)(s)
    n = Q.main_network(s)
    a = Q.action_network(n)
    v = fill(Q.value_network(n)[1], size(a))
    v + a .- mean(a)
end

mutable struct Policy{Q, OPT} <: AbstractPolicy
    ϵ             :: Float32
    ϵ_min         :: Float32
    ϵ_decay_steps :: Float32
    q_online      :: Q
    q_target      :: Q
    T             :: Int64
    optimizer     :: OPT
end
function Policy(ϵ, ϵ_min, ϵ_decay_steps, q)
    Policy(
        Float32(ϵ),
        Float32(ϵ_min),
        Float32(ϵ_decay_steps),
        q,
        deepcopy(q),
        0,
        RMSProp(0.000_5))
end

function policy_ϵ(p::Policy)
    @assert p.ϵ >= p.ϵ_min "ϵ must be greater than or equal to ϵ_min"
    decay_delta = p.ϵ - p.ϵ_min
    if decay_delta == 0 || p.ϵ_decay_steps <= 0
        p.ϵ_min
    else
        decay_base = (0.001/decay_delta)^(1/p.ϵ_decay_steps)
        ϵ = p.ϵ_min + decay_delta * decay_base^p.T
        @assert p.ϵ_min <= ϵ && ϵ <= p.ϵ
        ϵ
    end
end

function Reinforce.action(policy::Policy, r, s, A)
    ϵ = policy_ϵ(policy)
    policy.T += 1
    if rand() < ϵ
        rand(A)
    else
        argmax(policy.q_online(s)) |> network_i_to_env_a
    end
end

env_a_to_network_i(a) = a+1
network_i_to_env_a(a) = a-1

function polyak_average!(a, b, τ=0.01)
    for (pa, pb) in zip(a, b)
        pa .*= 1 - τ
        pa .+= pb * τ
    end
end

function optimize!(policy, sars₀, γ=1.0f0)
    γ = Float32(γ)
    sars = stack(sars₀, length(env.actions), env_a_to_network_i)
    a′ = mapslices(
        eavs -> onehot(argmax(eavs), length(eavs)),
        policy.q_online(sars.s′),
        dims=1)
    @assert typeof(a′) == Array{Float32, 2}
    qs′ = policy.q_target(sars.s′)
    @assert typeof(qs′) == Array{Float32, 2}
    target = sars.r + γ * sum(qs′ .* a′, dims=1) .* sars.f
    grads = gradient(params(policy.q_online)) do
        qs = policy.q_online(sars.s)
        predicted = sum(qs .* sars.a_hot, dims=1)
        mean((target .- predicted).^2)
    end
    update!(policy.optimizer, params(policy.q_online), grads)
    polyak_average!(params(policy.q_target), params(policy.q_online))
end

const env_step_limit = env.pyenv._max_episode_steps
const newline_frequency = 60

function run(episode_limit=10_000)
    sars = CircularBuffer{SARSF{Vector{Float32},Int8}}(50_000)
    q = Q()
    explore_policy = Policy(1.0, 0.3, 20_000, q)
    exploit_policy = Policy(0.0, 0.0, 0, q)
    explore_rewards = Float32[]
    exploit_rewards = Float32[]
    time_steps = 0
    start_time = now()
    newline_time = time() + newline_frequency
    try
        for episode in 1:episode_limit
            episode_t = 0
            explore_r = run_episode(env, explore_policy) do (s, a, r, s′)
                episode_t += 1
                @assert episode_t <= env_step_limit
                time_steps += 1
                failed = finished(env) && episode_t < env_step_limit
                push!(sars, SARSF(
                    Float32.(copy(s)),
                    Int8(a),
                    Float32(r),
                    Float32.(copy(s′)),
                    failed))
                if length(sars) > 64 * 5
                    optimize!(explore_policy, sample(sars, 64, replace=false))
                end
            end
            episode_t = 0
            exploit_r = run_episode(env, exploit_policy) do _
                episode_t += 1
                @assert episode_t <= env_step_limit
            end
            push!(explore_rewards, explore_r)
            push!(exploit_rewards, exploit_r)
            @printf("\u1b[?25l\u1b[0E%s ep %5d ts %6d, explore r %6.2f ± %6.2f, exploit r %6.2f ± %6.2f, pϵ %4.2f \u1b[0K\u1b[?25h",
                    Dates.format(Time(Nanosecond(now() - start_time)), "HH:MM:SS"),
                    episode, time_steps,
                    mean(last(explore_rewards, 100)), std(last(explore_rewards, 100)),
                    mean(last(exploit_rewards, 100)), std(last(exploit_rewards, 100)),
                    policy_ϵ(explore_policy))
            if time() >= newline_time
                newline_time += newline_frequency
                println()
            end
            if mean(last(exploit_rewards, 100)) >= 475; break end
        end
        @printf("\n------> Ran %5d episodes\n\n", length(explore_rewards))
    catch e
        if !isa(e, InterruptException); rethrow() end
        @printf("\n---> Interrupted at %d episodes\n\n", length(explore_rewards))
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
