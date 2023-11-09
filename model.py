import torch
import torch.nn as nn
import torch.nn.functional as F

DIAG_NUM = 1958
PROC_NUM = 1430
MED_NUM = 131
VOCABULARY_SIZE=429

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class MLP_mimic(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP_mimic, self).__init__()
        self.fc1=nn.Linear(input_size, 128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,output_size)

    def forward(self, gt_data_diagnose,gt_data_procedure):
        gt_data = torch.concat((gt_data_diagnose, gt_data_procedure), dim=1)
        out=F.relu(self.fc1(gt_data))
        out=out/out.sum()
        out=F.relu(self.fc2(out))
        out=self.fc3(out)
        return out
    

class TransformerEncoder_mimic(nn.Module):
    def __init__(self,input_size,embedding_size,output_size):
        super(TransformerEncoder_mimic, self).__init__()
        self.embeddings_diag= nn.Parameter(torch.randn(1, DIAG_NUM, embedding_size, requires_grad=True))
        self.embeddings_proc= nn.Parameter(torch.randn(1, PROC_NUM, embedding_size, requires_grad=True))

        encoder_layer=nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc1=nn.Linear(input_size,128)
        self.fc2=nn.Linear(128,output_size)
        
    def forward(self,gt_data_diagnose,gt_data_procedure):
        gt_data_diagnose=gt_data_diagnose.unsqueeze(dim=1)
        gt_data_procedure=gt_data_procedure.unsqueeze(dim=1)

        gt_data_diagnose=gt_data_diagnose.permute(0,2,1).mul(self.embeddings_diag)
        gt_data_procedure=gt_data_procedure.permute(0,2,1).mul(self.embeddings_proc)

        gt_data = torch.concat((gt_data_diagnose, gt_data_procedure), dim=1)
        gt_data=gt_data.permute(0,2,1)

        out = self.encoder(gt_data)
        out =torch.sum(out,dim=1)

        out = self.fc1(out)
        out=out/out.sum()
        out = F.relu(out)
        out = self.fc2(out)
        return out
    




class MLP_eicu(nn.Module):
    def __init__(self,input_size,embedding_size,output_size):
        super(MLP_eicu, self).__init__()
        self.fc1=nn.Linear(input_size, 128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,output_size)
        self.embedding= nn.Parameter(torch.randn(1 , 1, VOCABULARY_SIZE, embedding_size, requires_grad=True))

    def forward(self, disc_data,cont_data):
        disc_data=disc_data.unsqueeze(dim=2).permute(0,1,3,2)
        disc_data=disc_data.mul(self.embedding)
        disc_data = disc_data.flatten(2)
        gt_data = torch.cat((disc_data,cont_data),dim=-1)
        gt_data=gt_data.flatten(1)
        out=F.relu(self.fc1(gt_data))
        out=out/out.sum()
        out=F.relu(self.fc2(out))
        out=self.fc3(out)
        return out

    
 
class TransformerEncoder_eicu(nn.Module):
    def __init__(self,input_size,embedding_size,output_size):
        super(TransformerEncoder_eicu, self).__init__()
        self.embedding= nn.Parameter(torch.randn(1, VOCABULARY_SIZE, embedding_size, requires_grad=True))

        encoder_layer=nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc1=nn.Linear(input_size,128)
        self.fc2=nn.Linear(128,output_size)
        
    def forward(self, disc_data, cont_data):
        disc_data=disc_data.unsqueeze(dim=3)
        disc_data=disc_data.mul(self.embedding)
        disc_data = disc_data.flatten(2)
        gt_data = torch.cat((disc_data,cont_data),dim=-1)
        out = self.encoder(gt_data)
        out =torch.sum(out,dim=1)
        out = self.fc1(out)
        out=out/out.sum()
        out = F.relu(out)
        out = self.fc2(out)
        return out
    

