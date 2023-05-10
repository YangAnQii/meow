import torch

nz = 100
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
torch.manual_seed(1) # fix seed
fn = torch.randn(64, nz, 1, 1, device=device)
vaefn = torch.FloatTensor([64, 3, 20]).normal_()
#print(fixed_noise)