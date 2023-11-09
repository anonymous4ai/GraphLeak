# -*- coding: utf-8 -*-
import argparse
import torch
from graphleak import graphleak_mimic
from sklearn.metrics import roc_auc_score
from model import  MLP_mimic,TransformerEncoder_mimic,weights_init
from torch.utils.data import DataLoader
from utils import MimicDataset


parser = argparse.ArgumentParser(description='GrapgDLG on MIMIC-III Dataset')
parser.add_argument('--model', type=str, default="mlp",
                    help='the model used for training.')
parser.add_argument('--embedding-size', type=int, default="64",
                    help='the embdding size')
parser.add_argument('--batch-size', type=int, default="4",
                    help='the batch size')
parser.add_argument('--graph-prior', action='store_true', default=False,
                        help='wheather to use matrix normalization')
parser.add_argument('--graph-prior-med', action='store_true', default=True,
                        help='wheather to not use med matrix normalization')
parser.add_argument('--tag-loss', action='store_true', default=False,
                        help='wheather to use tag loss')
parser.add_argument('--w1', type=float, default="1e-5",
                    help='weight of diag')
parser.add_argument('--w2', type=float, default="1e-5",
                    help='weight of proc')
parser.add_argument('--w3', type=float, default="1e-5",
                    help='weight of med')
parser.add_argument('--wt', type=float, default="1e-2",
                    help='weight of tag')
parser.add_argument('--scale', type=int, default="5",
                    help='scale of konwledge graph')
parser.add_argument('--seed', type=int, default="1",
                    help='random seed')
args = parser.parse_args()


torch.manual_seed(args.seed)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

EMBEDDING_SIZE = args.embedding_size
DIAG_NUM = 1958
PROC_NUM = 1430
MED_NUM = 131
COUNT=100


print('#########################################')
print('Current model: '+args.model)
print('Embedding Size: '+ str(args.embedding_size)+'   Batch Size: '+ str(args.batch_size))
if args.graph_prior:
    print('Using '+str(args.scale)+'% Graph') 
    print('regularization coefficient: '+str(args.wd)+' '+str(args.wp)+' '+str(args.we))
if args.tag_loss:
    print('Using TAG Loss with coefficient ' + str(args.wt))
print('#########################################')


dataset=MimicDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=args.batch_size,
                          shuffle=True)
    

diag_auc_total,proc_auc_total,med_auc_total=0,0,0
for i, data in enumerate(train_loader):
    gt_diagnose_multihot, gt_procedure_multihot, gt_medicine_multihot = data
    gt_diagnose_multihot, gt_procedure_multihot, gt_medicine_multihot = gt_diagnose_multihot.to(device), gt_procedure_multihot.to(device), gt_medicine_multihot.to(device)
    
    
    dummy_diagnose_multihot = torch.randn(gt_diagnose_multihot.size()).to(device).requires_grad_(True)
    dummy_procedure_multihot =torch.randn(gt_procedure_multihot.size()).to(device).requires_grad_(True)
    dummy_medicine_multihot = torch.randn(gt_medicine_multihot.size()).to(device).requires_grad_(True)

    
    optimizer = torch.optim.Adam([dummy_diagnose_multihot,dummy_procedure_multihot,dummy_medicine_multihot])
    criterion = torch.nn.BCEWithLogitsLoss()
        

    if args.model == 'transformer':
        net = TransformerEncoder_mimic(input_size=DIAG_NUM+PROC_NUM,embedding_size=EMBEDDING_SIZE, output_size=MED_NUM).to(device)
        net.apply(weights_init)
    elif args.model == 'mlp':
        net = MLP_mimic(input_size=DIAG_NUM+PROC_NUM,output_size=MED_NUM).to(device)
        net.apply(weights_init)
    else:
        raise ValueError("no such model")
    

    pred = net(gt_diagnose_multihot,gt_procedure_multihot)
    y = criterion(pred, gt_medicine_multihot)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))


    dummy_diag_pred,dummy_proc_pred,dummy_med_pred = graphleak_mimic(dummy_diagnose_multihot,
                                                            dummy_procedure_multihot,
                                                            dummy_medicine_multihot,
                                                            original_dy_dx,optimizer,
                                                            net,criterion,args)
    dummy_diag_pred = torch.sigmoid(dummy_diag_pred)
    dummy_proc_pred = torch.sigmoid(dummy_proc_pred)
    dummy_med_pred = torch.sigmoid(dummy_med_pred)


    diagnose_auc=roc_auc_score(gt_diagnose_multihot.cpu().numpy(),dummy_diag_pred.detach().cpu().numpy(),average='micro')
    procedure_auc=roc_auc_score(gt_procedure_multihot.cpu().numpy(),dummy_proc_pred.detach().cpu().numpy(),average='micro')
    medicine_auc=roc_auc_score(gt_medicine_multihot.cpu().numpy(),dummy_med_pred.detach().cpu().numpy(),average='micro')
    print('Results of group '+str(i+1)+':',diagnose_auc,procedure_auc,medicine_auc,'\n')


    diag_auc_total+=diagnose_auc
    proc_auc_total+=procedure_auc
    med_auc_total+=medicine_auc
    print(str(i+1)+'-th cumulative round results:',diag_auc_total/(i+1),proc_auc_total/(i+1),med_auc_total/(i+1))


    if(i+1>=COUNT):
        print("End training, the total training data is "+str(i+1)+"group!")
        break


print('Final Result:',diag_auc_total/COUNT,proc_auc_total/COUNT,med_auc_total/COUNT)