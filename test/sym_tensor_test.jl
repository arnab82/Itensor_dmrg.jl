# Stage 1: block-sparse tensor core (QN, SymTensor) for the symmetry=true path.
#
# Everything is pinned against dense equivalents: a SymTensor round-trips through
# `dense`, block-sparse `contract`/`svd_truncated`/`qr_factorize` reproduce the
# dense contraction / SVD / QR, and truncation error equals the dropped weight.

const CH = NaiveDMRG
const SQN = NaiveDMRG.QN

# Random symmetric tensor: fill every flux-allowed block with noise.
function rand_sym(::Type{T}, legs::NTuple{R,<:Any}, flux) where {T,R}
    K = length(flux)
    blocks = Dict{NTuple{R,Int},Array{T,R}}()
    for key in CH._allowed_keys(legs, flux)
        blocks[key] = randn(T, CH._blockdims(legs, key))
    end
    return CH.SymTensor{T,R,K}(legs, flux, blocks)
end

# A physical (electron) leg and a couple of bond legs sharing K = 2 charges.
phys(arrow) = CH.SymIndex(arrow, [SQN(0, 0), SQN(1, 0), SQN(0, 1), SQN(1, 1)], [1, 1, 1, 1])

@testset verbose = true "Sym tensor core (Stage 1)" begin
    @testset "QN arithmetic" begin
        a, b = SQN(1, 0), SQN(0, 1)
        @test (a + b).q == (1, 1)
        @test (a - b).q == (1, -1)
        @test (-a).q == (-1, 0)
        @test (2 * a).q == (2, 0)
        @test zero(SQN(0, 0)).q == (0, 0)
        @test SQN(0, 1) < SQN(1, 0)                 # deterministic ordering
    end

    @testset "dense round-trip and flux enforcement" begin
        bl = CH.SymIndex(+1, [SQN(0, 0), SQN(1, 0)], [1, 2])
        br = CH.SymIndex(-1, [SQN(0, 0), SQN(1, 0), SQN(0, 1), SQN(1, 1), SQN(2, 0)], [1, 2, 1, 1, 1])
        legs = (bl, phys(+1), br)
        A = rand_sym(ComplexF64, legs, SQN(0, 0))
        @test CH.numblocks(A) > 0
        @test CH.dense(CH.SymTensor(CH.dense(A), legs, SQN(0, 0))) ≈ CH.dense(A)
        # a generic (non-symmetric) dense array has weight in forbidden blocks
        junk = randn(ComplexF64, size(CH.dense(A)))
        @test_throws ArgumentError CH.SymTensor(junk, legs, SQN(0, 0))
    end

    @testset "contract matches dense (matrices)" begin
        a = CH.SymIndex(-1, [SQN(0, 0), SQN(1, 0), SQN(0, 1)], [2, 2, 1])
        b = CH.SymIndex(+1, [SQN(0, 0), SQN(1, 0), SQN(0, 1)], [1, 3, 2])
        c = CH.SymIndex(+1, [SQN(0, 0), SQN(1, 0), SQN(0, 1)], [2, 1, 2])
        A = rand_sym(Float64, (a, b), SQN(0, 0))
        B = rand_sym(Float64, (CH.dual(b), c), SQN(0, 0))
        C = CH.contract(A, B, (2,), (1,))
        @test CH.dense(C) ≈ CH.dense(A) * CH.dense(B)
    end

    @testset "contract matches dense (rank-3 MPS bond)" begin
        bl = CH.SymIndex(+1, [SQN(0, 0), SQN(1, 0)], [1, 2])
        br = CH.SymIndex(-1, [SQN(0, 0), SQN(1, 0), SQN(0, 1), SQN(1, 1), SQN(2, 0), SQN(2, 1)],
                         [1, 1, 1, 1, 1, 1])
        br2 = CH.SymIndex(-1, [SQN(a, b) for a in 0:3 for b in 0:2], fill(1, 12))
        A = rand_sym(ComplexF64, (bl, phys(+1), br), SQN(0, 0))
        B = rand_sym(ComplexF64, (CH.dual(br), phys(+1), br2), SQN(0, 0))
        C = CH.contract(A, B, (3,), (1,))                  # join on the shared bond
        Ad, Bd = CH.dense(A), CH.dense(B)
        l, s, m = size(Ad); _, t, r = size(Bd)             # C[l,s,t,r] = Σ_m A·B
        Cref = reshape(reshape(Ad, l * s, m) * reshape(Bd, m, t * r), l, s, t, r)
        @test CH.dense(C) ≈ Cref
    end

    @testset "svd_truncated reconstructs and reports discarded weight" begin
        bl = CH.SymIndex(+1, [SQN(0, 0), SQN(1, 0), SQN(0, 1)], [2, 3, 2])
        br = CH.SymIndex(-1, [SQN(a, b) for a in 0:2 for b in 0:2], fill(2, 9))
        A = rand_sym(Float64, (bl, phys(+1), br), SQN(0, 0))

        # untruncated: U·S·V == A, no discarded weight
        U, S, V, disc = CH.svd_truncated(A, (1, 2), (3,); maxdim=10_000, cutoff=0.0)
        Arec = CH.contract(U, CH.absorb_S_right(S, V), (3,), (1,))
        @test CH.dense(Arec) ≈ CH.dense(A)
        @test disc ≈ 0 atol = 1e-9
        # U is an isometry: stacking all row sectors, columns are orthonormal
        Ud = reshape(CH.dense(U), :, CH.totaldim(U.legs[3]))
        @test Ud' * Ud ≈ I(size(Ud, 2)) atol = 1e-9

        # truncated: ‖A - Arec‖² equals the reported discarded weight
        Ut, St, Vt, disct = CH.svd_truncated(A, (1, 2), (3,); maxdim=4, cutoff=0.0)
        Arect = CH.contract(Ut, CH.absorb_S_right(St, Vt), (3,), (1,))
        @test sum(abs2, CH.dense(A) .- CH.dense(Arect)) ≈ disct atol = 1e-7
        @test CH.totaldim(Ut.legs[3]) <= 4
    end

    @testset "qr_factorize reconstructs and gives an isometry" begin
        bl = CH.SymIndex(+1, [SQN(0, 0), SQN(1, 0)], [2, 2])
        br = CH.SymIndex(-1, [SQN(a, b) for a in 0:2 for b in 0:1], fill(2, 6))
        A = rand_sym(ComplexF64, (bl, phys(+1), br), SQN(0, 0))
        Q, Rf = CH.qr_factorize(A, (1, 2), (3,))
        Arec = CH.contract(Q, Rf, (3,), (1,))
        @test CH.dense(Arec) ≈ CH.dense(A)
        Qd = reshape(CH.dense(Q), :, CH.totaldim(Q.legs[3]))
        @test Qd' * Qd ≈ I(size(Qd, 2)) atol = 1e-9
    end
end
