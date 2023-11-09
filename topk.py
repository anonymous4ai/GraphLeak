import torch
import argparse
from torch.utils.data import DataLoader
from utils import MimicDataset,EicuDataset,to_multihot
from sklearn.metrics import roc_auc_score

DIAG_NUM = 1958
PROC_NUM = 1430
MED_NUM = 131
VOCABULARY_SIZE=429

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

parser = argparse.ArgumentParser(description='TopK baseline')
parser.add_argument('--batch-size',type=int,default=1)
parser.add_argument('--name',type=str,default='eicu')
args=parser.parse_args()

if args.name == 'mimic':
    diagnoise=torch.zeros(DIAG_NUM)
    procedure=torch.zeros(PROC_NUM)
    medicine=torch.zeros(MED_NUM)
    len_d_total=0
    len_p_total=0
    len_m_total=0

    mimic=MimicDataset()
    mimic_loader = DataLoader(dataset=mimic,batch_size=args.batch_size,shuffle=False)
    count = 0 
    for _, data in enumerate(mimic_loader):
        gt_diagnose_multihot, gt_procedure_multihot, gt_medicine_multihot = data
        len_d_total += sum(gt_diagnose_multihot[0])
        len_p_total += sum(gt_procedure_multihot[0])
        len_m_total += sum(gt_medicine_multihot[0])
        count+=1
        for j,k in enumerate(gt_diagnose_multihot[0]):
            if k == 1:
                diagnoise[j]+=1
        for j,k in enumerate(gt_procedure_multihot[0]):
            if k == 1:
                procedure[j]+=1
        for j,k in enumerate(gt_medicine_multihot[0]):
            if k == 1:
                medicine[j]+=1
    len_d_avg=torch.round(len_d_total/count)
    len_p_avg=torch.round(len_p_total/count)
    len_m_avg=torch.round(len_m_total/count)

    _,index_d=torch.topk(diagnoise,int(len_d_avg))
    _,index_p=torch.topk(procedure,int(len_p_avg))
    _,index_m=torch.topk(medicine,int(len_m_avg))

    diagnoise_baseline = to_multihot(index_d.unsqueeze(0),DIAG_NUM)
    procedure_baseline = to_multihot(index_p.unsqueeze(0),PROC_NUM)
    medicine_baseline = to_multihot(index_m.unsqueeze(0),MED_NUM)
    
    diag_auc_total,proc_auc_total,med_auc_total,count=0,0,0,0
    for _, data in enumerate(mimic_loader):
        gt_diagnose_multihot, gt_procedure_multihot, gt_medicine_multihot = data
        diagnose_auc=roc_auc_score(gt_diagnose_multihot.cpu().numpy(),diagnoise_baseline.cpu().numpy(),average='micro')
        procedure_auc=roc_auc_score(gt_procedure_multihot.cpu().numpy(),procedure_baseline.cpu().numpy(),average='micro')
        medicine_auc=roc_auc_score(gt_medicine_multihot.cpu().numpy(),medicine_baseline.cpu().numpy(),average='micro')
        diag_auc_total+=diagnose_auc
        proc_auc_total+=procedure_auc
        med_auc_total+=medicine_auc
        count+=1
    print('Final Results:',diag_auc_total/count,proc_auc_total/count,med_auc_total/count)


if args.name == 'eicu':
    dataset=EicuDataset(train=True)
    train_loader = DataLoader(dataset=dataset,batch_size=args.batch_size,shuffle=False)

    disc=torch.zeros(VOCABULARY_SIZE)
    len_total=0
    count=0
    count_1=0
    count_0=0
    for _, data in enumerate(train_loader):
        disc_data,cont_data,label=data
        if label[0][0]==1:
            count_0+=1
        else:
            count_1+=1
        for visit in disc_data[0]:
            len_total+=sum(visit)
            count+=1
            for j,k in enumerate(visit):
                if k == 1:
                    disc[j]+=1
    len_avg=torch.round(len_total/count)
    _,index=torch.topk(disc,int(len_avg))
    baseline = to_multihot(index.unsqueeze(0),VOCABULARY_SIZE).squeeze(0)
    auc_total,count=0,0
    for _, data in enumerate(train_loader):
        disc_data,cont_data,label=data
        for visit in disc_data[0]:
            auc=roc_auc_score(visit.cpu().numpy(),baseline.cpu().numpy(),average='micro')
            auc_total+=auc
            count+=1
    print('Final Results:',auc_total/count,count_0/(count_0+count_1))
        
