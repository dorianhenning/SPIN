import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class MSE_Var_Loss(nn.Module):
    def __init__(self):
        super(MSE_Var_Loss, self).__init__()

    def forward(self, mean, var, gt):
        # var = JT * diag(var) * J
        #pdb.set_trace()
        n = var.shape[1]
        bs = var.shape[0]
        delta = torch.zeros(gt.shape).cuda()
        loss = torch.zeros(bs).cuda()
        for i in range(n):
            # iterating over keypoints
            
            delta[:,i,:] = gt[:,i,:] - mean[:,i,:]
            
            expm = torch_expm(var[:,i,:,:])

            loss_1 = torch.bmm(expm, delta[:,i,:].clone().unsqueeze(2))
            loss_1 = torch.bmm(delta[:,i,:].clone().unsqueeze(1), loss_1).squeeze()
            #loss_1 = torch.mul(torch.transpose(delta, dim0=1, dim1=2), torch.bmm(expm, delta))
            #print(var.shape[0])
            loss_2 = torch.det(var[:,i,:,:])

            loss += .5 * (loss_1 + loss_2) / n

        return loss.mean()

## BACKUP: works for betas
#class MSE_Var_Loss(nn.Module):
#    def __init__(self):
#        super(MSE_Var_Loss, self).__init__()
#
#    def forward(self, mean, var, gt):
#        # var = JT * diag(var) * J
#        pdb.set_trace()
#            
#        delta = gt - mean
#        
#        expm = torch_expm(var)
#
#        loss_1 = torch.bmm(expm, delta.view(-1, size, 1))
#        loss_1 = torch.bmm(delta.view(-1, 1, size), loss_1)
#        #loss_1 = torch.mul(torch.transpose(delta, dim0=1, dim1=2), torch.bmm(expm, delta))
#        #print(var.shape[0])
#        loss_2 = torch.det(var)
#
#        loss = .5 * (loss_1 + loss_2)
#
#        return loss.mean()

#%%
def torch_expm(A):
    """ """
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1,2), keepdim=True))
    
    # Scaling step
    maxnorm = torch.Tensor([5.371920351148152]).type(A.dtype).to(A.device)
    zero = torch.Tensor([0.0]).type(A.dtype).to(A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    Ascaled = A / 2.0**n_squarings    
    n_squarings = n_squarings.flatten().type(torch.int32)
    
    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V
    R, _ = torch.solve(P, Q) # solve P = Q*R
    
    # Unsquaring step
    expmA = [ ]
    for i in range(n_A):
        l = [R[i]]
        for _ in range(n_squarings[i]):
            l.append(l[-1].mm(l[-1]))
        expmA.append(l[-1])
    
    return torch.stack(expmA)

#%%
def torch_log2(x):
    return torch.log(x) / torch.log(torch.Tensor([2.0])).type(x.dtype).to(x.device)

#%%    
def torch_pade13(A):
    b = torch.Tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.]).type(A.dtype).to(A.device)
        
    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A,A)
    A4 = torch.matmul(A2,A2)
    A6 = torch.matmul(A4,A2)
    U = torch.matmul(A, torch.matmul(A6, b[13]*A6 + b[11]*A4 + b[9]*A2) + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = torch.matmul(A6, b[12]*A6 + b[10]*A4 + b[8]*A2) + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    return U, V
