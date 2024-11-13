using ITensors
using Printf
# some sample tensors
i = Index(2, "i")
j = Index(3, "j")
k = Index(4, "k")

A = randomITensor(i, j)
B = randomITensor(j, k)

# Method 1: Using the * operator
C1 = A * B

# Method 2: Using the contract function
C2 = contract(A, B)

# Both C1 and C2 should be equivalent

# Contracting multiple tensors
l = Index(2, "l")
E = randomITensor(k, l)

F = A * B * E  # This contracts over the common indices automatically

# Or use reduce with *
G = reduce(*, [A, B, E])

# Partial contractions
H = randomITensor(i, j, k)
I = randomITensor(j, k, l)
j=contract(H,  I)  
k=H * I
@assert(j == k)
k_=inds(k)
# Print the results to verify
println("C1 = ", C1)
println("C2 = ", C2)
println("F = ", F)
println("G = ", G)
println("H = ", H)
println("I = ", I)
println("j = ", j)
println("k = ", k)
