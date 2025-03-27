import torch


def net_params_to_tensor(net, flatten=False) -> torch.Tensor:
    # flat_params = [ar.to_numpy(param) for param in net.parameters()]
    params = [param for param in net.parameters()]
    if flatten:
        flat_params = [torch.flatten(param) for param in net.parameters()]
        return torch.concat(flat_params)
    return params
        
def net_grads_to_tensor(net, clip=False, flatten = True) -> torch.Tensor:
    param_grads = []
    if clip:
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    for param in net.parameters():
        if param.grad is not None:
            # Clone to avoid modifying the original tensor
            if flatten:
                param_grads.append(param.grad.data.clone().view(-1))
            else:
                param_grads.append(param.grad.data.clone())
    if flatten:
        param_grads = torch.cat(param_grads)
    return param_grads