## Some functions are from:
## https://github.com/zhanghang1989/PyTorch-Encoding

"""Encoding Package Core NN Modules."""
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F

def scaled_l2(X, C, S):
    """
    scaled_l2 distance
    Args:
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
        S (k):      scale cofficient
    Return:
        D (b*n*k):  relative distance to each code
    Note:
        apparently the X^2 + C^2 - 2XC computation is 2x faster than
        elementwise sum, perhaps due to friendly cache in gpu
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert S.numel() == C.shape[0], "scale, codeword num mismatch"
    """
    # simplier but slower
    X = X.unsqueeze(2)
    C = C[None, None,...]
    norm = torch.norm(X-C, dim=-1).pow(2.0)
    scaled_norm = S * norm
    """
    b, n, d = X.shape
    X = X.view(-1, d)  # [bn, d]
    Ct = C.t()  # [d, k]
    X2 = X.pow(2.0).sum(-1, keepdim=True)  # [bn, 1]
    C2 = Ct.pow(2.0).sum(0, keepdim=True)  # [1, k]
    norm = X2 + C2 - 2.0 * X.mm(Ct)  # [bn, k]
    scaled_norm = S * norm
    D = scaled_norm.view(b, n, -1)  # [b, n, k]
    return D


def aggregate(A, X, C):
    """
    aggregate residuals from N samples
    Args:
        A (b*n*k):  weight of each feature contribute to code residual
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
    Return:
        E (b*k*d):  residuals to each code
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert A.shape[:2] == X.shape[:2], "weight, input dim mismatch"
    X = X.unsqueeze(2)  # [b, n, d] -> [b, n, 1, d]
    C = C[None, None, ...]  # [k, d] -> [1, 1, k, d]
    A = A.unsqueeze(-1)  # [b, n, k] -> [b, n, k, 1]
    R = (X - C) * A  # [b, n, k, d]
    E = R.sum(dim=1)  # [b, k, d]
    return E

class Encoding3D(Module):
    def __init__(self, C, K):
        super(Encoding3D, self).__init__()
        self.C, self.K = C, K
        self.codewords = Parameter(torch.Tensor(K, C), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()
        self.drop = torch.nn.Dropout3d(p=0.1)

    def reset_params(self):
        std1 = 1./((self.K*self.C)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 5D tensor
        assert(X.size(1) == self.C)
        B, C, D, H, W = X.size()
        if X.dim() == 3:
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            X = X.view(B, self.C, -1).transpose(1, 2).contiguous()
        elif X.dim() == 5:
            X = X.view(B, self.C, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
        coefA = A.transpose(1, 2).view(B, self.K, D, H, W).contiguous()
        # add dropout
        coefA = self.drop(coefA)

        # aggregate
        E = aggregate(A, X, self.codewords)
        return E, coefA

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + '=>' + str(self.K) + 'x' \
            + str(self.C) + ')'
