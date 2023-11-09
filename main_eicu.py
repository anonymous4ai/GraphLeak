import torch
import torch.nn as nn
import argparse
from utils import EicuDataset
from torch.utils.data import DataLoader
from model import MLP_eicu,TransformerEncoder_eicu,weights_init
from graphleak import graphleak_eicu
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='GrapgDLG on eICU Dataset')
parser.add_argument('--model', type=str, default="transformer",
                    help='the model used for training.')
parser.add_argument('--embedding-size', type=int, default="5",
                    help='the embdding size')
parser.add_argument('--batch-size', type=int, default="1",
                    help='the batch size')
parser.add_argument('--graph-prior', action='store_true',default=False,
                        help='matrix normalize')
parser.add_argument('--w1', type=float, default="1e-5",
                    help='weight')
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
VOCABULARY_SIZE=429
VISIT_SIZE=50
CONTINUOUS_SIZE=13
COUNT=100


print('#########################################')
print('Current model: '+args.model)
print('Embedding Size: '+ str(args.embedding_size)+'   Batch Size: '+ str(args.batch_size))
if args.graph_prior:
    print('Using '+str(args.scale)+'% Graph') 
    print('regularization coefficient: '+str(args.we))
if args.tag_loss:
    print('Using TAG Loss with coefficient ' + str(args.wt))
print('#########################################')


dataset=EicuDataset(train=True)
train_loader = DataLoader(dataset=dataset,
                          batch_size=args.batch_size,
                          shuffle=True)


data_auc_total,data_mse_total,label_auc_total = 0,0,0
for i ,data in enumerate(train_loader):
    disc_data,cont_data,label=data
    disc_data,cont_data,label = disc_data.to(device),cont_data.to(device),label.to(device)
    

    dummy_disc_data = torch.randn(disc_data.size()).to(device).requires_grad_(True)
    dummy_cont_data = torch.randn(cont_data.size()).to(device).requires_grad_(True)
    dummy_label =torch.randn(label.size()).to(device).requires_grad_(True)


    optimizer = torch.optim.Adam([dummy_disc_data,dummy_cont_data,dummy_label])
    criterion = torch.nn.BCEWithLogitsLoss()
    

    if args.model == 'transformer':
        net = TransformerEncoder_eicu(input_size=VOCABULARY_SIZE*EMBEDDING_SIZE+CONTINUOUS_SIZE,embedding_size=EMBEDDING_SIZE,output_size=2).to(device)
        net.apply(weights_init)
    elif args.model == 'mlp':
        net = MLP_eicu(input_size =(VOCABULARY_SIZE*EMBEDDING_SIZE+CONTINUOUS_SIZE)*VISIT_SIZE,embedding_size=EMBEDDING_SIZE,output_size = 2).to(device)
        net.apply(weights_init)
    else:
        raise ValueError("no such model, only mlp and transformer")


    pred = net(disc_data,cont_data)
    y = criterion(pred, label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))


    dummy_disc_data,dummy_cont_data,dummy_label = graphleak_eicu(dummy_disc_data,
                                                                dummy_cont_data,
                                                                dummy_label,
                                                                original_dy_dx,optimizer,
                                                                net,criterion,args)
    
    
    dummy_disc_data=torch.sigmoid(dummy_disc_data)
    dummy_label=torch.sigmoid(dummy_label)
    
    
    data_auc=0
    for disc_data_visit,dummy_disc_data_visit in zip(disc_data,dummy_disc_data):
        data_auc_visit=roc_auc_score(disc_data_visit.cpu().numpy(),dummy_disc_data_visit.detach().cpu().numpy(),average='micro')
        data_auc+=data_auc_visit
    data_auc /= len(disc_data)
    data_mse = nn.MSELoss(reduction='mean')(dummy_cont_data,cont_data).detach().cpu().numpy()
    label_auc=roc_auc_score(label.cpu().numpy(),dummy_label.detach().cpu().numpy(),average='micro')
    print('Results of group '+str(i+1)+':',data_auc,data_mse,label_auc,'\n')


    data_auc_total+=data_auc
    data_mse_total+=data_mse
    label_auc_total+=label_auc
    print(str(i+1)+'-th cumulative round results:',data_auc_total/(i+1),data_mse_total/(i+1),label_auc_total/(i+1))


    if(i+1==COUNT):
        print("End training, the total training data is "+str(i+1)+"group!")
        break


print('Final Result:',data_auc_total/COUNT,data_mse_total/COUNT,label_auc_total/COUNT)
    