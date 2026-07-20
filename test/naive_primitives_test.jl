using Random

function reference_absorb_left(L, A, W)
    out = zeros(promote_type(eltype(L), eltype(A), eltype(W)),
                size(A, 3), size(W, 4), size(A, 3))
    for rb in axes(A, 3), b in axes(W, 4), rk in axes(A, 3),
        lb in axes(A, 1), a in axes(W, 1), lk in axes(A, 1),
        so in axes(A, 2), si in axes(A, 2)
        out[rb, b, rk] += conj(A[lb, so, rb]) * L[lb, a, lk] *
                          W[a, so, si, b] * A[lk, si, rk]
    end
    return out
end

function reference_absorb_right(R, A, W)
    out = zeros(promote_type(eltype(R), eltype(A), eltype(W)),
                size(A, 1), size(W, 1), size(A, 1))
    for lb in axes(A, 1), a in axes(W, 1), lk in axes(A, 1),
        so in axes(A, 2), rb in axes(A, 3), si in axes(A, 2),
        b in axes(W, 4), rk in axes(A, 3)
        out[lb, a, lk] += conj(A[lb, so, rb]) * W[a, so, si, b] *
                          A[lk, si, rk] * R[rb, b, rk]
    end
    return out
end

function left_block(psi, stop)
    d = psi.d
    block = ones(ComplexF64, 1, 1)
    for site in 1:stop
        A = psi.tensors[site]
        next = zeros(ComplexF64, size(block, 1) * d, size(A, 3))
        for prefix in axes(block, 1), l in axes(A, 1), s in 1:d, r in axes(A, 3)
            row = prefix + (s - 1) * size(block, 1)
            next[row, r] += block[prefix, l] * A[l, s, r]
        end
        block = next
    end
    return block
end

function right_block(psi, start)
    d = psi.d
    block = ones(ComplexF64, 1, 1)
    for site in psi.N:-1:start
        A = psi.tensors[site]
        next = zeros(ComplexF64, size(A, 1), d * size(block, 2))
        for l in axes(A, 1), s in 1:d, r in axes(A, 3), suffix in axes(block, 2)
            col = s + (suffix - 1) * d
            next[l, col] += A[l, s, r] * block[r, suffix]
        end
        block = next
    end
    return block
end

function two_site_embedding(psi, site)
    d = psi.d
    left = left_block(psi, site - 1)
    right = right_block(psi, site + 2)
    shape = (size(left, 2), d, d, size(right, 1))
    embedding = zeros(ComplexF64, d^psi.N, prod(shape))
    left_size = size(left, 1)
    right_sites = psi.N - site - 1

    for prefix in axes(left, 1), l in axes(left, 2), s1 in 1:d, s2 in 1:d,
        r in axes(right, 1), suffix in axes(right, 2)
        row = prefix + (s1 - 1) * left_size + (s2 - 1) * left_size * d +
              (suffix - 1) * left_size * d^2
        col = LinearIndices(shape)[l, s1, s2, r]
        embedding[row, col] += left[prefix, l] * right[r, suffix]
    end
    @assert size(right, 2) == d^right_sites
    return embedding, shape
end

@testset verbose = true "NaiveDMRG contraction primitives against dense references" begin
    C = NaiveDMRG
    rng = MersenneTwister(20240718)
    H = C.heisenberg_mpo(4; J=0.7, hz=0.2)
    psi = C.random_MPS(4, 2, 4; rng=rng)
    cache = C.environments(H, psi)

    @testset "left and right environments" begin
        L = ones(ComplexF64, 1, 1, 1)
        @test cache.left[1] == L
        for site in 1:psi.N
            L = reference_absorb_left(L, psi.tensors[site], H.tensors[site])
            @test cache.left[site + 1] ≈ L atol=1e-12
        end

        R = ones(ComplexF64, 1, 1, 1)
        @test cache.right[psi.N + 1] == R
        for site in psi.N:-1:1
            R = reference_absorb_right(R, psi.tensors[site], H.tensors[site])
            @test cache.right[site] ≈ R atol=1e-12
        end

        dense_energy = real(dot(C.dense(psi), C.dense(H) * C.dense(psi)))
        @test cache.left[end][] ≈ dense_energy atol=1e-12
        @test cache.right[1][] ≈ dense_energy atol=1e-12
    end

    @testset "two-site effective Hamiltonian" begin
        for site in 1:psi.N-1
            embedding, shape = two_site_embedding(psi, site)
            dense_effective = embedding' * C.dense(H) * embedding
            x = randn(rng, ComplexF64, prod(shape))
            projected = C.effective_action(
                x,
                cache.left[site],
                H.tensors[site],
                H.tensors[site + 1],
                cache.right[site + 2],
                shape,
            )
            @test projected ≈ dense_effective * x atol=2e-11
            buffered = similar(projected)
            @test C.effective_action!(
                buffered, x, cache.left[site], H.tensors[site],
                H.tensors[site + 1], cache.right[site + 2], shape
            ) === buffered
            @test buffered ≈ projected atol=2e-11
            @test dense_effective ≈ dense_effective' atol=2e-11
        end
    end


    @testset "minimal sweep cache" begin
        sweep_cache = C.right_sweep_cache(H, psi)
        @test sweep_cache.left[1] == ones(ComplexF64, 1, 1, 1)
        for cut in 3:psi.N+1
            @test sweep_cache.right[cut] ≈ cache.right[cut] atol=1e-12
        end
    end
end
