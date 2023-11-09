import torch
import numpy as np
from torch.utils.data import Dataset

VOCABULARY_SIZE=429
DIAG_NUM = 1958
PROC_NUM = 1430
MED_NUM = 131


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


class MimicDataset(Dataset):
    def __init__(self, train=True):
        if train:
            self.diagnose_data = torch.load('data/mimic/diagnose.pth')
            self.procedure_data = torch.load('data/mimic/procedure.pth')
            self.medication_data = torch.load('data/mimic/medicine.pth')
        else:
            self.diagnose_data = torch.load('data/mimic/diagnose_test.pth')
            self.procedure_data = torch.load('data/mimic/procedure_test.pth')
            self.medication_data = torch.load('data/mimic/medicine_test.pth')
    def __getitem__(self, index):
        return self.diagnose_data[index], self.procedure_data[index], self.medication_data[index]

    def __len__(self):
        return len(self.diagnose_data)


class EicuDataset(Dataset):
    def __init__(self,train=True):
        if train:
            self.train_x_disc = np.load('data/eicu/X_train_disc.npy')
            self.train_x_disc = torch.tensor(self.train_x_disc).to(torch.float32)
            self.train_x_cont = np.load('data/eicu/X_train_cont.npy')
            self.train_x_cont = torch.tensor(self.train_x_cont).to(torch.float32)
            self.train_y = np.load('data/eicu/Y_train_onehot.npy')
            self.train_y = torch.tensor(self.train_y).to(torch.float32)
        else:
            self.train_x_disc = np.load('data/eicu/X_test_disc.npy')
            self.train_x_disc = torch.tensor(self.train_x_disc).to(torch.float32)
            self.train_x_cont = np.load('data/eicu/X_test_cont.npy')
            self.train_x_cont = torch.tensor(self.train_x_cont).to(torch.float32)
            self.train_y = np.load('data/eicu/Y_test_onehot.npy')
            self.train_y = torch.tensor(self.train_y).to(torch.float32)
    def __getitem__(self, index):
        return  self.train_x_disc[index],self.train_x_cont[index],self.train_y[index]
    
    def __len__(self):
        return len(self.train_x_disc)


def to_multihot(target, num_classes=131, normalize=False):
    multihot_target = torch.zeros(target.shape[0], num_classes, device=target.device)
    count=0
    for i,row in zip(range(len(target)),target):
        for _,col in zip(range(len(row)),row):
            multihot_target[i][int(col.item())]=1
            count+=1
    if normalize:
        return multihot_target/count
    else:
        return multihot_target


def pre_prossess_data_mimic(data,train=True):
    diagnose_data = []
    procedure_data = []
    medication_data = []
    if train==True:
        for i in range(0,3000):
            for j in range(len(data[i])):
                diagnose_data.append(to_multihot(torch.Tensor(data[i][j][0]).unsqueeze(dim=0).to(device).to(torch.int64),num_classes=DIAG_NUM).squeeze())
                procedure_data.append(to_multihot(torch.Tensor(data[i][j][1]).unsqueeze(dim=0).to(device).to(torch.int64),num_classes=PROC_NUM).squeeze())
                print(data[i][j][2])
                medication_data.append(to_multihot(torch.Tensor(data[i][j][2]).unsqueeze(dim=0).to(device).to(torch.int64),num_classes=MED_NUM).squeeze())
    else:
        for i in range(3000,len(data)):
            for j in range(len(data[i])):
                diagnose_data.append(to_multihot(torch.Tensor(data[i][j][0]).unsqueeze(dim=0).to(device).to(torch.int64),num_classes=DIAG_NUM).squeeze())
                procedure_data.append(to_multihot(torch.Tensor(data[i][j][1]).unsqueeze(dim=0).to(device).to(torch.int64),num_classes=PROC_NUM).squeeze())
                medication_data.append(to_multihot(torch.Tensor(data[i][j][2]).unsqueeze(dim=0).to(device).to(torch.int64),num_classes=MED_NUM).squeeze())
    return diagnose_data, procedure_data, medication_data


def pre_prossess_data_eicu():
    train_x = np.load('data/X_train.npy')
    train_x_disc = train_x[:,:50,:7]
    train_x_cont = train_x[:,:50,7:]
    print(train_x_disc.shape)

    data1 = []
    for datai in train_x_disc:
        data2=[]
        for dataj in datai:
            print(dataj)
            multihot=np.zeros([429])
            for i in dataj:
                i = int(i)
                multihot[i]=1
            data2.append(multihot)
        data2=np.array(data2)
        data1.append(data2)
    data1=np.array(data1)
    np.save('data/X_train_disc.npy',data1)
    np.save('data/X_train_cont.npy',train_x_cont)


def build_occurrence(list1,list2,matrix):
    for i in list1:
        for j in list2:
            matrix[int(i)][int(j)]+=1
    return matrix


def build_graph_mimic(train_loader):
    
    diag_diag_matrix=torch.zeros(DIAG_NUM,DIAG_NUM)
    proc_proc_matrix=torch.zeros(PROC_NUM,PROC_NUM)
    med_med_matrix=torch.zeros(MED_NUM,MED_NUM)
    
    for i, data in enumerate(train_loader):
        gt_data_diagnose, gt_data_procedure, gt_label = data
        print(gt_data_diagnose)
        gt_data_diagnose = torch.Tensor(
            gt_data_diagnose).to(device).to(torch.int64).tolist()
        gt_data_procedure = torch.Tensor(
            gt_data_procedure).to(device).to(torch.int64).tolist()
        gt_label = torch.Tensor(
            gt_label).to(device).to(torch.int64).tolist()
        print(gt_data_diagnose)

        diag_diag_matrix=build_occurrence(gt_data_diagnose,gt_data_diagnose,diag_diag_matrix)
        proc_proc_matrix=build_occurrence(gt_data_procedure,gt_data_procedure,proc_proc_matrix)
        med_med_matrix=build_occurrence(gt_label,gt_label,med_med_matrix)
    
    return diag_diag_matrix,proc_proc_matrix,med_med_matrix


def build_graph_eicu():
    data=np.load('data/eicu/X_train.npy')
    count=int(len(data)*0.5)
    data=data[:,:50,:7]
    data_matrix=torch.zeros([VOCABULARY_SIZE,VOCABULARY_SIZE])
    i=0
    for datai in data:
        i+=1
        if i>count:
            break
        for dataj in datai:
            data_matrix=build_occurrence(dataj,dataj,data_matrix)
    data_matrix /= data_matrix.sum()
    torch.save(data_matrix, "data/eicu/eicu_matrix50.pth")