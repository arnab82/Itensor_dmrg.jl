using ITensors

"""
    svd_truncate!(psi::MPS, b::Int, new_tensor::ITensor)
    args:



"""
function svd_truncate(psi::MPS, b::Int, phi::ITensor; maxdim::Int, cutoff::Float64, normalize::Bool, ortho::String)
    if ortho == "left"
        orthocenter = b
        # Use only the indices of psi[b] for the left tensor
        U_inds = (inds(psi[b])...,)
    elseif ortho == "right"
        orthocenter = b + 1
        # Use only the indices of psi[b+1] for the right tensor
        U_inds = (inds(psi[b+1])...,)
    else
        error("Invalid ortho option: $ortho. Use 'left' or 'right'.")
    end

    # Perform SVD
    U, S, V = svd(phi, U_inds; maxdim=maxdim, cutoff=cutoff)

    # Update the MPS tensors based on sweep direction
    if ortho == "left"
        psi[b] = U
        psi[b+1] = S * V
    else  # ortho == "right"
        psi[b] = U * S
        psi[b+1] = V
    end

    # Normalize the updated MPS if requested
    if normalize
        normalize!(psi)
    end

    return psi
end