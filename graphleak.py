import torch
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


def graphleak_mimic(dummy_diagnose_multihot,
            dummy_procedure_multihot,
            dummy_medicine_multihot,
            original_dy_dx,optimizer,
            net,criterion,args):
    

    if args.scale == 100:
        diag_matrix=torch.load('data/mimic/diag_diag_matrix_norm100.pth').to(device)
        proc_matrix=torch.load('data/mimic/proc_proc_matrix_norm100.pth').to(device)
    elif args.scale == 5:
        diag_matrix=torch.load('data/mimic/diag_diag_matrix_norm5.pth').to(device)
        proc_matrix=torch.load('data/mimic/proc_proc_matrix_norm5.pth').to(device)
    elif args.scale == 10:
        diag_matrix=torch.load('data/mimic/diag_diag_matrix_norm10.pth').to(device)
        proc_matrix=torch.load('data/mimic/proc_proc_matrix_norm10.pth').to(device)
    elif args.scale == 20:
        diag_matrix=torch.load('data/mimic/diag_diag_matrix_norm20.pth').to(device)
        proc_matrix=torch.load('data/mimic/proc_proc_matrix_norm20.pth').to(device)
    elif args.scale == 50:
        diag_matrix=torch.load('data/mimic/diag_diag_matrix_norm50.pth').to(device)
        proc_matrix=torch.load('data/mimic/proc_proc_matrix_norm50.pth').to(device)
    else:
        raise ValueError("no such scale, only 5,10,20,50,100")
    med_matrix=torch.load('data/mimic/med_med_matrix_norm.pth').to(device)
    

    for iters in range(30000):
        def closure():
            optimizer.zero_grad()
            dummy_diagnose=torch.sigmoid(dummy_diagnose_multihot)
            dummy_procedure=torch.sigmoid(dummy_procedure_multihot)
            dummy_medicine=torch.sigmoid(dummy_medicine_multihot)


            dummy_pred = net(dummy_diagnose,dummy_procedure)
            dummy_loss = criterion(dummy_pred, dummy_medicine)
            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True)


            grad_diff = 0
            # DLG loss
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
                # TAG loss
                if args.tag_loss:
                    tag_loss=torch.abs(gx - gy).sum()
                    grad_diff += args.wt * tag_loss
                    
            # Reg loss
            if args.graph_prior:
                loss_diag = (dummy_diagnose @ diag_matrix @ dummy_diagnose.T).sum()
                loss_proc = (dummy_procedure @ proc_matrix @ dummy_procedure.T).sum()
                if args.graph_prior_med:
                    grad_diff = grad_diff - args.w1 * loss_diag - args.w2 * loss_proc
                else:
                    loss_med = (dummy_medicine @ med_matrix @ dummy_medicine.T).sum()
                    grad_diff = grad_diff - args.w1 * loss_diag - args.w2 * loss_proc - args.w3 * loss_med
            grad_diff.backward()
            return grad_diff
        
        optimizer.step(closure)
        if iters % 5000 == 0:
            current_loss = closure()
            print(str(iters)+' iters Loss:', "%.4f" % current_loss.item())
    return dummy_diagnose_multihot,dummy_procedure_multihot,dummy_medicine_multihot


def graphleak_eicu(dummy_disc_data,
            dummy_cont_data,
            dummy_label,
            original_dy_dx,optimizer,
            net,criterion,args):
    

    if args.scale==100:
        eicu_matrix=torch.load('data/eicu/eicu_matrix100.pth').to(device)
    elif args.scale==5:
        eicu_matrix=torch.load('data/eicu/eicu_matrix5.pth').to(device)
    elif args.scale==10:
        eicu_matrix=torch.load('data/eicu/eicu_matrix10.pth').to(device)
    elif args.scale==20:
        eicu_matrix=torch.load('data/eicu/eicu_matrix20.pth').to(device)
    elif args.scale==50:
        eicu_matrix=torch.load('data/eicu/eicu_matrix50.pth').to(device)
    else:
        raise ValueError("no such scale, only 5,10,20,50,100")
    eicu_matrix=eicu_matrix.unsqueeze(dim=0)


    for iters in range(30000):
        def closure():
            optimizer.zero_grad()
            dummy_disc_data_norm=torch.sigmoid(dummy_disc_data)
            dummy_pred = net(dummy_disc_data_norm,dummy_cont_data)
            dummy_label_norm=torch.sigmoid(dummy_label)
            dummy_loss = criterion(dummy_pred,dummy_label_norm)
            dummy_dy_dx = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            # DLG loss
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
                # TAG loss
                if args.tag_loss:
                    tag_loss=torch.abs(gx - gy).sum()
                    grad_diff += args.wt * tag_loss
            # Reg loss
            if args.graph_prior:
                loss_kg = (dummy_disc_data_norm @ eicu_matrix @ dummy_disc_data_norm.permute(0,2,1)).sum()
                grad_diff = grad_diff - args.w1 * loss_kg
            grad_diff.backward()
            return grad_diff
        
        optimizer.step(closure)
        if iters % 5000 == 0:
            current_loss = closure()
            print(str(iters)+' iters Loss:', "%.4f" % current_loss.item())
    return dummy_disc_data,dummy_cont_data,dummy_label