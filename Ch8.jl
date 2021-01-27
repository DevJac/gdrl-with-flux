using Flux
using Flux.Optimise: update!
using OpenAIGym
using Plots
using ProgressMeter
using Statistics

const env = GymEnv(:CartPole, :v1)

struct Q{T}
    network :: T
end
Flux.@functor Q
function Q()
    network = Chain(
        Dense(length(env.state), 512, relu),
        Dense(512, 128, relu),
        Dense(128, length(env.actions), identity))
    Q(network)
end

@info("Q parameter count", params(Q()) .|> length |> sum)

struct Policy <: AbstractPolicy
    ϵ :: Float32
    q_network :: Q
end

function Reinforce.action(policy::Policy, r, s, A)
    if rand() < policy.ϵ
        return rand(A)
    else
        return argmax(policy.q_network.network(s))-1
    end
end

struct SARSF{S, A}
    s  :: S
    a  :: A
    r  :: Float32
    s′ :: S
    f  :: Bool
end

const opt = RMSProp(0.000_5)

function update(q, sars, k=40)
    all_s = convert(Array{Float32,2}, reduce(hcat, map(x -> copy(x.s), sars)))
    all_s′ = reduce(hcat, map(x -> copy(x.s′), sars))
    all_a = reduce(hcat, map(x -> x.a == 0 ? [1.0, 0.0] : [0.0, 1.0], sars))
    all_r = reduce(hcat, map(x -> x.r, sars))
    all_f = reduce(hcat, map(x -> x.f ? 0.0 : 1.0, sars))
    prms = params(q)
    for k in 1:k
        target = all_r + 1.0 * (maximum(q.network(all_s′), dims=1)) .* all_f
        grads = gradient(prms) do
            qs = q.network(all_s)
            predicted = sum(qs .* all_a, dims=1)
            mean((predicted .- target).^2)
        end
        update!(opt, prms, grads)
    end
end

const batch_size = 1024

function run(n_episodes)
    sars = SARSF[]
    exploit_rewards = Float32[]
    explore_rewards = Float32[]
    q = Q()
    explore_policy = Policy(0.5, q)
    exploit_policy = Policy(0.0, q)
    progress = Progress(n_episodes)
    try
        for episode in 1:n_episodes
            T = 0
            explore_r = run_episode(env, explore_policy) do (s, a, r, s′)
                T += 1
                @assert T <= 500
                failed = finished(env) && T < 500
                push!(sars, SARSF(s, a, Float32(r), s′, failed))
                if length(sars) >= batch_size
                    update(q, sars)
                    empty!(sars)
                end
            end
            exploit_r = run_episode(env, exploit_policy) do _
                @assert T <= 500
            end
            push!(explore_rewards, explore_r)
            push!(exploit_rewards, exploit_r)
            next!(progress; showvalues = [(:explore_rewards, mean(last(explore_rewards, 100))),
                                          (:exploit_rewards, mean(last(exploit_rewards, 100)))])
            if mean(last(exploit_rewards, 100)) >= 475; break end
        end
    finally
        close(env)
    end
    (exploit_policy, exploit_rewards, explore_rewards)
end

last(xs, n) = xs[max(1, end-n+1):end]

function demo_policy(policy, n_episodes=20)
    try
        for _ in 1:n_episodes
            run_episode(env, policy) do _
                render(env)
            end
        end
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
# p, r, e = run(5000)
# graph(r)
# demo_policy(p)
