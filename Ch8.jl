using Dates
using Flux
using Flux.Optimise: update!
using OpenAIGym
using Plots
using Printf
using Statistics

const env = GymEnv(:CartPole, :v1)

struct Q{T}
    network :: T
end
Flux.@functor Q
function Q()
    network = Chain(
        Dense(length(env.state), 512, relu,
              initW=Flux.kaiming_uniform, initb=Flux.kaiming_uniform),
        Dense(512, 128, relu,
              initW=Flux.kaiming_uniform, initb=Flux.kaiming_uniform),
        Dense(128, length(env.actions), identity,
              initW=Flux.kaiming_uniform, initb=Flux.kaiming_uniform))
    Q(network)
end

@info("Q parameter count", params(Q()) .|> length |> sum)

network_action_to_env_action(a) = a-1
env_action_to_network_action(a) = a+1

struct EGreedyPolicy <: AbstractPolicy
    ϵ :: Float32
    q_network :: Q
end

function Reinforce.action(policy::EGreedyPolicy, r, s, A)
    if rand() < policy.ϵ
        return rand(A)
    else
        return argmax(policy.q_network.network(s)) |> network_action_to_env_action
    end
end

struct SARSF{S, A}
    s       :: S
    a       :: A
    r       :: Float32
    s′      :: S
    failed  :: Bool
end

function onehot(hot_index, size)
    result = zeros(Float32, size)
    result[hot_index] = 1.0f0
    result
end

const opt = RMSProp(0.000_5)

function optimize!(q, sars, epochs=40, γ=1.0f0)
    γ = Float32(γ)
    s = reduce(hcat, map(x -> x.s, sars))
    a = reduce(hcat, map(x -> onehot(env_action_to_network_action(x.a), length(env.actions)), sars))
    r = reduce(hcat, map(x -> x.r, sars))
    s′ = reduce(hcat, map(x -> x.s′, sars))
    f = reduce(hcat, map(x -> x.failed ? 0.0f0 : 1.0f0, sars))
    @assert typeof(s) == Array{Float32, 2}
    @assert typeof(a) == Array{Float32, 2}
    @assert typeof(r) == Array{Float32, 2}
    @assert typeof(s′) == Array{Float32, 2}
    @assert typeof(f) == Array{Float32, 2}
    @assert typeof(γ) == Float32
    for _ in 1:epochs
        target = r + γ * maximum(q.network(s′), dims=1) .* f
        grads = gradient(params(q)) do
            qs = q.network(s)
            predicted = sum(qs .* a, dims=1)
            mean((predicted .- target).^2)
        end
        update!(opt, params(q), grads)
    end
end

const batch_size = 1024
const episode_step_limit = env.pyenv._max_episode_steps
const newline_frequency = 60

@info("env step limit", episode_step_limit)

function run(episode_limit=20000)
    sars = SARSF{Vector{Float32},Int8}[]
    exploit_rewards = Float32[]
    explore_rewards = Float32[]
    q = Q()
    explore_policy = EGreedyPolicy(0.5, q)
    exploit_policy = EGreedyPolicy(0.0, q)
    time_steps = 0
    start_time = now()
    newline_time = time() + newline_frequency
    try
        for episode in 1:episode_limit
            T = 0
            explore_r = run_episode(env, explore_policy) do (s, a, r, s′)
                time_steps += 1
                T += 1
                @assert T <= episode_step_limit
                failed = finished(env) && T < episode_step_limit
                push!(sars, SARSF(Float32.(copy(s)), Int8(a), Float32(r), Float32.(copy(s′)), failed))
                if length(sars) >= batch_size
                    optimize!(q, sars)
                    empty!(sars)
                end
            end
            exploit_r = run_episode(env, exploit_policy) do _
                @assert T <= episode_step_limit
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

# To run in the REPL, include this file, then:
# p, r, e = run()
# graph(r)
# demo_policy(p)
