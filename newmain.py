import os
from pyexpat import model
import torch
import argparse
import warnings
import numpy as np
from torch_geometric.loader import dataloader
from utils import get_split, SVMEvaluator
from tqdm import trange, tqdm

torch.set_printoptions(threshold=np.inf)

from poolings.CONNet import CONNet
from models.YHNet import YHNet, Config 
from models.HGCL2 import HGCL2, Config

head_path = os.getcwd()
import test  # new
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
import random
from evalute_embedding import svc, linearsvc

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# from evalute_embedding import evaluate_embedding

from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--model', type=str, default="CONPool", help='model name')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')  # 学习率
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')  # 权重衰减率
parser.add_argument('--dataset', type=str, default='PTC_MR', help='DD/NCI1/NCI109/MUTAG/PROTEINS')
parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=20, help='path to save result')

args = parser.parse_args()


def test1(args, encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(args.device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        g, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)

    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
    return result["accuracy"], result["std"]


# def linearsvc(embeds, labels):
#     x = embeds.detach().cpu().numpy()
#     y = labels.detach().cpu().numpy()
#     params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
#     kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
#     accuracies = []
#     for train_index, test_index in kf.split(x, y):
#         x_train, x_test = x[train_index], x[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
#         classifier.fit(x_train, y_train)
#         accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
#     return np.mean(accuracies), np.std(accuracies)


def test2(args, encoder_model, dataloader):
    encoder_model.eval()
    # x = []
    # y = []
    # for data in dataloader:
    #     data = data.to(args.device)
    #     if data.x is None:
    #         num_nodes = data.batch.size(0)
    #         data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
    #     _, g = encoder_model(data.x, data.edge_index, data.batch, args.device)
    #     x.append(g)
    #     y.append(data.y)

    # x = torch.cat(x, dim=0)
    # y = torch.cat(y, dim=0)
    #x , y = encoder_model.get_embeddings(args.device,dataloader)
    #x , y = encoder_model.get_single_embeddings(args.device,dataloader)
    x , y = encoder_model.conv1.get_embeddings(args.device, dataloader)
    acc_mean, acc_std = svc(x, y)
    print('Before training: acc_mean = ', acc_mean, '  acc_std = ', acc_std)
    with open('randomDD2_result.txt', 'a') as f:
        f.write(str(acc_mean) + ' ')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def sup_train(args,epochs,  model, train_loader):
    log_interval = 5
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, epochs + 1 ):
        loss_train = 0.0

        correct = 0
        for i, data in enumerate(train_loader): 
            optimizer.zero_grad()
            data = data.to(args.device)

            _, output = model(data.x, data.edge_index, data.batch, args.device)
            cls_loss = torch.nn.functional.nll_loss(output, data.y)
            
            loss = cls_loss 
            loss.backward()
            optimizer.step()
            loss_train += cls_loss.item()
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()

        if epoch % log_interval == 0:
            model.eval()
            x, y= model.get_embeddings(args.device, train_loader)
            acc_mean, acc_std = svc(x, y)
            print('acc_mean = ', acc_mean, '  acc_std = ', acc_std)
            with open('427mutag.txt', 'a') as f:
                 f.write(str(acc_mean) + ' ')




def con_train(args, epochs, dataset, seed, layers, batch, model, dataloader, loadertest, mode):
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)  # 仿照GraphCL里面的设置
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log_interval = 5
    # x1, x2, x3, x4, y1, y2, y3, y4 = [],[],[],[],[],[],[],[]
    pbar = tqdm(range(1,epochs + 1),ncols=100)
    for epoch in pbar:#trange(1, epochs + 1):
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _, _= model(data.x, data.edge_index, data.batch, args.device)
            #loss = model.compute_gl_loss()
            
            # loss =   model.compute_inner()  +  3 * model.compute_hier()  之前大部分都是1+3 #model.compute_inner() + model.compute_hier() #+ model.compute_Farloss() #model.compute_gl_loss()   #+ model.compute_Farloss() # + model.compute_Farloss()#model.compute_gl_loss()  0.1 * model.compute_gl_loss()# model.compute_H() #model.compute_YH_loss()  #+ model.compute_conloss() #+ h_loss
            if mode == '0':
                loss = model.compute_inner()
            elif mode == '0.25':
                loss =  0.75 * model.compute_inner() + 0.25 * model.compute_hier() 
            elif mode == '0.5':
                loss = (model.compute_inner()  +  model.compute_hier())/2
            elif mode == '0.75':
                loss = 0.25 * model.compute_inner() + 0.75 * model.compute_hier() 
            elif mode == '1':
                loss = model.compute_hier()
            # 返回对比损失
            dis_loss = loss #0.01 * loss #0.01 * loss  # 10 * h_loss
            dis_loss.backward()

            # for para in model.pool1.parameters():
            #     print(para)
            # for para in model.parameters():
            #     print(para)
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
        if epoch % log_interval ==0 :
            model.eval()
            x, y= model.get_embeddings(args.device,dataloader)
            # x = []
            # y = []
            # for data in dataloader:
            #     data = data.to(args.device)
            #     g, _ = model(data.x, data.edge_index, data.batch, args.device)
            #     x.append(g)
            #     y.append(data.y)

            # x = torch.cat(x, dim=0)
            # y = torch.cat(y, dim=0)
            acc_mean, acc_std = svc(x, y)
            print('acc_mean = ', acc_mean, '  acc_std = ', acc_std)
            with open('522.txt', 'a') as f:
                 f.write(str(acc_mean) + ' ')

    with open('522.txt', 'a') as f:
        f.write('\n')


def con_train_topk(args, epochs, dataset, seed, layers, batch, model, dataloader):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _, _ = model(data.x, data.edge_index, data.batch, args.device)

            loss = model.compute_YH_loss()
            # 返回对比损失
            dis_loss = loss  # 10 * h_loss
            dis_loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))


def train_pool(args, epochs, dataset, seed, layers, batch, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr =0.01)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_all = 0.0
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(args.device)
            _, _= model(data.x, data.edge_index, data.batch, args.device)
            loss = model.compute_Farloss()
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))


def main():
    seeds = [0,1,2,3,4]
    batch = [256] #12 8,256,512  # [32, 64, 128, 256]
    epoch = [5]  # [ 10, 20, 40]
    layers = [2]  # [2, 3, 4]
    modes = ['0.5']
    # FAR= [3]
    ds = ['REDDIT-BINARY']#['PROTEINS','COLLAB','NCI1','REDDIT-BINARY','DD']#['IMDB-BINARY','PTC_MR','PROTEINS','IMDB-MULTI','MUTAG']#['MUTAG', 'NCI1', 'PTC_MR', 'IMDB-MULTI','IMDB-BINARY','PROTEINS', 'COLLAB', 'DD','REDDIT-BINARY']#['MUTAG','IMDB-MULTI','MUTAG','IMDB-BINARY','NCI1','PROTEINS','PTC_MR','COLLAB','DD','REDDIT-BINARY'] #,'IMDB-BINARY', 'COLLAB']# 'PTC_MR', 'IMDB-BINARY', 'DD', 'REDDIT-BINARY', 'IMDB-MULTI', ]  # ['COLLAB', 'IMDB-BINARY', 'MUTAG','IMDB-MULTI', 'PROTEINS', 'COLLAB']
    for d in ds:
        print(f'-------------------------------{d}------------------------------')
        for l in layers:
            for b in batch:
                # for f in FAR:
                for m in modes:
                    for e in epoch:
                        for i in range(1):
                            if torch.cuda.is_available():
                                args.device = "cuda:1"
                            else:
                                args.device = "cpu"
                            # OK
                            seed = seeds[i]
                            set_seed(seed)
                            # OK
                            print(f'Dataset: {d}, Layer:{l}, Batch: {b}, Epoch: {e}, seed: {seed}, Far:{3}, mode:{m} ')
                            loader, loader_test = test.build_loader(args, b, d)  # 后面改
                            config = Config()
                            model = YHNet(config, args, l, 32).to(args.device)
                            #model = HGCL1(config, args, l, 32).to(args.device)
                            #test2(args, model, loader)
                            
                            train_pool(args, 3, d, seed, l, b, model, loader)
                            con_train(args, e, d, seed, l, b, model, loader, loader_test, m)

                            if i == 4:
                                with open('522.txt', 'a') as f:
                                    f.write('\n')
                                with open('randomDD2_result.txt', 'a') as f:
                                    f.write('\n')
                            pass 

if __name__ == '__main__':
    main()
datasetMUTAG,batch64,seed0,mode0.5,loss4.531389236450195
datasetMUTAG,batch64,seed0,mode0.5,loss4.2724283536275225
datasetMUTAG,batch64,seed0,mode0.5,loss4.166045983632405
datasetMUTAG,batch64,seed0,mode0.5,loss4.113980134328206
datasetMUTAG,batch64,seed0,mode0.5,loss4.087818145751953
datasetMUTAG,batch64,seed0,mode0.5,loss4.076595624287923
datasetMUTAG,batch64,seed0,mode0.5,loss4.057734330495198
datasetMUTAG,batch64,seed0,mode0.5,loss4.0454972585042315
datasetMUTAG,batch64,seed0,mode0.5,loss4.015971819559733
datasetMUTAG,batch64,seed0,mode0.5,loss4.015947262446086
datasetMUTAG,batch64,seed0,mode0.5,loss3.999939282735189
datasetMUTAG,batch64,seed0,mode0.5,loss3.983515818913778
datasetMUTAG,batch64,seed0,mode0.5,loss3.98282782236735
datasetMUTAG,batch64,seed0,mode0.5,loss3.970906893412272
datasetMUTAG,batch64,seed0,mode0.5,loss3.9693525632222495
datasetMUTAG,batch64,seed0,mode0.5,loss3.9796554247538247
datasetMUTAG,batch64,seed0,mode0.5,loss3.955054680506388
datasetMUTAG,batch64,seed0,mode0.5,loss3.9598964850107827
datasetMUTAG,batch64,seed0,mode0.5,loss3.9477411905924478
datasetMUTAG,batch64,seed0,mode0.5,loss3.9577131271362305
datasetMUTAG,batch64,seed0,mode0.5,loss3.9455296993255615
datasetMUTAG,batch64,seed0,mode0.5,loss3.942468802134196
datasetMUTAG,batch64,seed0,mode0.5,loss3.9309732913970947
datasetMUTAG,batch64,seed0,mode0.5,loss3.9214231173197427
datasetMUTAG,batch64,seed0,mode0.5,loss3.9218706289927163
datasetMUTAG,batch64,seed0,mode0.5,loss3.924257198969523
datasetMUTAG,batch64,seed0,mode0.5,loss3.918996016184489
datasetMUTAG,batch64,seed0,mode0.5,loss3.9111501375834146
datasetMUTAG,batch64,seed0,mode0.5,loss3.9017066955566406
datasetMUTAG,batch64,seed0,mode0.5,loss3.8975592454274497
datasetMUTAG,batch64,seed0,mode0.5,loss3.8903278509775796
datasetMUTAG,batch64,seed0,mode0.5,loss3.8911868731180825
datasetMUTAG,batch64,seed0,mode0.5,loss3.88993247350057
datasetMUTAG,batch64,seed0,mode0.5,loss3.8955350716908774
datasetMUTAG,batch64,seed0,mode0.5,loss3.909595807393392
datasetMUTAG,batch64,seed0,mode0.5,loss3.901444673538208
datasetMUTAG,batch64,seed0,mode0.5,loss3.8926824728647866
datasetMUTAG,batch64,seed0,mode0.5,loss3.8839504718780518
datasetMUTAG,batch64,seed0,mode0.5,loss3.9060919284820557
datasetMUTAG,batch64,seed0,mode0.5,loss3.897791067759196
datasetPTC_MR,batch64,seed0,mode0.5,loss4.191976070404053
datasetPTC_MR,batch64,seed0,mode0.5,loss3.9682772954305015
datasetPTC_MR,batch64,seed0,mode0.5,loss3.8878918488820395
datasetPTC_MR,batch64,seed0,mode0.5,loss3.848694682121277
datasetPTC_MR,batch64,seed0,mode0.5,loss3.8367530504862466
datasetPTC_MR,batch64,seed0,mode0.5,loss3.829663117726644
datasetPTC_MR,batch64,seed0,mode0.5,loss3.8182700475056968
datasetPTC_MR,batch64,seed0,mode0.5,loss3.8128032286961875
datasetPTC_MR,batch64,seed0,mode0.5,loss3.809878865877787
datasetPTC_MR,batch64,seed0,mode0.5,loss3.8000086545944214
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7889685233434043
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7939126094182334
datasetPTC_MR,batch64,seed0,mode0.5,loss3.78346053759257
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7762518723805747
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7720650831858316
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7707058986028037
datasetPTC_MR,batch64,seed0,mode0.5,loss3.767141064008077
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7585081259409585
datasetPTC_MR,batch64,seed0,mode0.5,loss3.763274073600769
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7617273728052774
datasetPTC_MR,batch64,seed0,mode0.5,loss3.755401690800985
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7510958115259805
datasetPTC_MR,batch64,seed0,mode0.5,loss3.750805139541626
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7528151671091714
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7388755877812705
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7427796125411987
datasetPTC_MR,batch64,seed0,mode0.5,loss3.745646595954895
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7432123819986978
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7379097938537598
datasetPTC_MR,batch64,seed0,mode0.5,loss3.73841122786204
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7301055192947388
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7285452683766684
datasetPTC_MR,batch64,seed0,mode0.5,loss3.726664145787557
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7274797757466636
datasetPTC_MR,batch64,seed0,mode0.5,loss3.72689696153005
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7214901447296143
datasetPTC_MR,batch64,seed0,mode0.5,loss3.7290501594543457
datasetPTC_MR,batch64,seed0,mode0.5,loss3.721114675203959
datasetPTC_MR,batch64,seed0,mode0.5,loss3.719424525896708
datasetPTC_MR,batch64,seed0,mode0.5,loss3.724409341812134
datasetPROTEINS,batch64,seed0,mode0.5,loss4.112111065122816
datasetPROTEINS,batch64,seed0,mode0.5,loss4.007939683066474
datasetPROTEINS,batch64,seed0,mode0.5,loss3.9799656867980957
datasetPROTEINS,batch64,seed0,mode0.5,loss3.969834009806315
datasetPROTEINS,batch64,seed0,mode0.5,loss3.947272645102607
datasetPROTEINS,batch64,seed0,mode0.5,loss3.938546299934387
datasetPROTEINS,batch64,seed0,mode0.5,loss3.936559107568529
datasetPROTEINS,batch64,seed0,mode0.5,loss3.9322566058900623
datasetPROTEINS,batch64,seed0,mode0.5,loss3.9203578763537936
datasetPROTEINS,batch64,seed0,mode0.5,loss3.9141658676995172
datasetPROTEINS,batch64,seed0,mode0.5,loss3.9137530061933727
datasetPROTEINS,batch64,seed0,mode0.5,loss3.9099779393937855
datasetPROTEINS,batch64,seed0,mode0.5,loss3.9006291230519614
datasetPROTEINS,batch64,seed0,mode0.5,loss3.905693689982096
datasetPROTEINS,batch64,seed0,mode0.5,loss3.905029058456421
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8940224250157676
datasetPROTEINS,batch64,seed0,mode0.5,loss3.890615807639228
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8919441435072155
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8821831809149847
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8841812213261924
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8838451041115656
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8828910324308605
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8782516585456
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8781976567374334
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8724384042951794
datasetPROTEINS,batch64,seed0,mode0.5,loss3.867769241333008
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8701731628841824
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8648620314068265
datasetPROTEINS,batch64,seed0,mode0.5,loss3.856879022386339
datasetPROTEINS,batch64,seed0,mode0.5,loss3.855656941731771
datasetPROTEINS,batch64,seed0,mode0.5,loss3.859113759464688
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8633268541759915
datasetPROTEINS,batch64,seed0,mode0.5,loss3.856754846043057
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8513015773561268
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8457184897528753
datasetPROTEINS,batch64,seed0,mode0.5,loss3.840391066339281
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8445880280600653
datasetPROTEINS,batch64,seed0,mode0.5,loss3.8435837427775064
datasetPROTEINS,batch64,seed0,mode0.5,loss3.844636254840427
datasetPROTEINS,batch64,seed0,mode0.5,loss3.837751626968384
datasetNCI1,batch64,seed0,mode0.5,loss4.03586252285884
datasetNCI1,batch64,seed0,mode0.5,loss3.939524635901818
datasetNCI1,batch64,seed0,mode0.5,loss3.9317178029280444
datasetNCI1,batch64,seed0,mode0.5,loss3.9175762726710395
datasetNCI1,batch64,seed0,mode0.5,loss3.9173588862785924
datasetNCI1,batch64,seed0,mode0.5,loss3.9110465013063873
datasetNCI1,batch64,seed0,mode0.5,loss3.906290439458994
datasetNCI1,batch64,seed0,mode0.5,loss3.8989661693573
datasetNCI1,batch64,seed0,mode0.5,loss3.8929458691523626
datasetNCI1,batch64,seed0,mode0.5,loss3.8859876339252177
datasetNCI1,batch64,seed0,mode0.5,loss3.8807060315058783
datasetNCI1,batch64,seed0,mode0.5,loss3.8773007869720457
datasetNCI1,batch64,seed0,mode0.5,loss3.8731606630178597
datasetNCI1,batch64,seed0,mode0.5,loss3.8698457901294416
datasetNCI1,batch64,seed0,mode0.5,loss3.8670142540564902
datasetNCI1,batch64,seed0,mode0.5,loss3.8629459674541766
datasetNCI1,batch64,seed0,mode0.5,loss3.8639370551476113
datasetNCI1,batch64,seed0,mode0.5,loss3.8582251878885123
datasetNCI1,batch64,seed0,mode0.5,loss3.860047065294706
datasetNCI1,batch64,seed0,mode0.5,loss3.85726501758282
datasetNCI1,batch64,seed0,mode0.5,loss3.855023431777954
datasetNCI1,batch64,seed0,mode0.5,loss3.8445159728710467
datasetNCI1,batch64,seed0,mode0.5,loss3.8278564856602597
datasetNCI1,batch64,seed0,mode0.5,loss3.8207278911884015
datasetNCI1,batch64,seed0,mode0.5,loss3.818857658826388
datasetNCI1,batch64,seed0,mode0.5,loss3.8170266188108
datasetNCI1,batch64,seed0,mode0.5,loss3.8170846902407134
datasetNCI1,batch64,seed0,mode0.5,loss3.8146448979010947
datasetNCI1,batch64,seed0,mode0.5,loss3.8135196208953857
datasetNCI1,batch64,seed0,mode0.5,loss3.8143211071307843
datasetNCI1,batch64,seed0,mode0.5,loss3.812811569067148
datasetNCI1,batch64,seed0,mode0.5,loss3.8100899622990534
datasetNCI1,batch64,seed0,mode0.5,loss3.808279694043673
datasetNCI1,batch64,seed0,mode0.5,loss3.8085376849541297
datasetNCI1,batch64,seed0,mode0.5,loss3.8059464601370006
datasetNCI1,batch64,seed0,mode0.5,loss3.805050802230835
datasetNCI1,batch64,seed0,mode0.5,loss3.805167245864868
datasetNCI1,batch64,seed0,mode0.5,loss3.8057090979356034
datasetNCI1,batch64,seed0,mode0.5,loss3.80515209711515
datasetNCI1,batch64,seed0,mode0.5,loss3.804435502565824
datasetCOLLAB,batch64,seed0,mode0.5,loss4.096391682271604
datasetCOLLAB,batch64,seed0,mode0.5,loss3.9790444329932884
datasetCOLLAB,batch64,seed0,mode0.5,loss3.946581968554744
datasetCOLLAB,batch64,seed0,mode0.5,loss3.961451693817421
datasetCOLLAB,batch64,seed0,mode0.5,loss3.9273005017527827
datasetCOLLAB,batch64,seed0,mode0.5,loss3.919594133341754
datasetCOLLAB,batch64,seed0,mode0.5,loss3.9037893259966814
datasetCOLLAB,batch64,seed0,mode0.5,loss3.9076585990411266
datasetCOLLAB,batch64,seed0,mode0.5,loss3.895554529296027
datasetCOLLAB,batch64,seed0,mode0.5,loss3.901502105924818
datasetCOLLAB,batch64,seed0,mode0.5,loss3.916140772678234
datasetCOLLAB,batch64,seed0,mode0.5,loss3.888372288809882
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8790077854085854
datasetCOLLAB,batch64,seed0,mode0.5,loss3.876682718594869
datasetCOLLAB,batch64,seed0,mode0.5,loss3.87457095252143
datasetCOLLAB,batch64,seed0,mode0.5,loss3.863134997862357
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8580365490030357
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8548520229480885
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8552958214724504
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8503248073436596
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8414117715976857
datasetCOLLAB,batch64,seed0,mode0.5,loss3.850068697222957
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8464860871986106
datasetCOLLAB,batch64,seed0,mode0.5,loss3.845968586427194
datasetCOLLAB,batch64,seed0,mode0.5,loss3.841899598086322
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8524300257364907
datasetCOLLAB,batch64,seed0,mode0.5,loss3.850956576841849
datasetCOLLAB,batch64,seed0,mode0.5,loss3.844140710654082
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8387858558584145
datasetCOLLAB,batch64,seed0,mode0.5,loss3.833781198219017
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8249416572076305
datasetCOLLAB,batch64,seed0,mode0.5,loss3.831289949240508
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8320069445504084
datasetCOLLAB,batch64,seed0,mode0.5,loss3.824741999308268
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8311281469133167
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8221406230220087
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8259919219546847
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8217519963229143
datasetCOLLAB,batch64,seed0,mode0.5,loss3.822103155983819
datasetCOLLAB,batch64,seed0,mode0.5,loss3.8189088194458574
datasetDD,batch64,seed0,mode0.5,loss4.119682575526991
datasetDD,batch64,seed0,mode0.5,loss3.9439931041315983
datasetDD,batch64,seed0,mode0.5,loss3.897187358454654
datasetDD,batch64,seed0,mode0.5,loss3.8699443842235364
datasetDD,batch64,seed0,mode0.5,loss3.8619895357834664
datasetDD,batch64,seed0,mode0.5,loss3.856386510949386
datasetDD,batch64,seed0,mode0.5,loss3.8417673110961914
datasetDD,batch64,seed0,mode0.5,loss3.839109182357788
datasetDD,batch64,seed0,mode0.5,loss3.830962017962807
datasetDD,batch64,seed0,mode0.5,loss3.8288530048571134
datasetDD,batch64,seed0,mode0.5,loss3.8289057330081335
datasetDD,batch64,seed0,mode0.5,loss3.8165910494954964
datasetDD,batch64,seed0,mode0.5,loss3.8166230226817883
datasetDD,batch64,seed0,mode0.5,loss3.812318023882414
datasetDD,batch64,seed0,mode0.5,loss3.812097198084781
datasetDD,batch64,seed0,mode0.5,loss3.8148457502063953
datasetDD,batch64,seed0,mode0.5,loss3.8020782721670052
datasetDD,batch64,seed0,mode0.5,loss3.806558558815404
datasetDD,batch64,seed0,mode0.5,loss3.803643038398341
datasetDD,batch64,seed0,mode0.5,loss3.8070742080086157
datasetDD,batch64,seed0,mode0.5,loss3.8010966903284977
datasetDD,batch64,seed0,mode0.5,loss3.794559077212685
datasetDD,batch64,seed0,mode0.5,loss3.7969633654544226
datasetDD,batch64,seed0,mode0.5,loss3.792530950747038
datasetDD,batch64,seed0,mode0.5,loss3.8037540159727397
datasetDD,batch64,seed0,mode0.5,loss3.7931845941041646
datasetDD,batch64,seed0,mode0.5,loss3.794794885735763
datasetDD,batch64,seed0,mode0.5,loss3.782976552059776
datasetDD,batch64,seed0,mode0.5,loss3.785943959888659
datasetDD,batch64,seed0,mode0.5,loss3.7803834739484286
datasetDD,batch64,seed0,mode0.5,loss3.7822014407107702
datasetDD,batch64,seed0,mode0.5,loss3.7877240431936166
datasetDD,batch64,seed0,mode0.5,loss3.785078249479595
datasetDD,batch64,seed0,mode0.5,loss3.783979064539859
datasetDD,batch64,seed0,mode0.5,loss3.782297548494841
datasetDD,batch64,seed0,mode0.5,loss3.7853468593798185
datasetDD,batch64,seed0,mode0.5,loss3.7776848516966166
datasetDD,batch64,seed0,mode0.5,loss3.774353215568944
datasetDD,batch64,seed0,mode0.5,loss3.7724960101278207
datasetDD,batch64,seed0,mode0.5,loss3.777232245395058
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss4.361299142241478
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss4.125498756766319
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss4.109689004719257
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss4.02734762430191
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss4.005758337676525
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9186970070004463
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.907662384212017
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9478696286678314
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.962112106382847
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.923975296318531
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8922683224081993
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8872891515493393
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.915188819169998
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9244848042726517
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8662141636013985
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.898460015654564
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9134974405169487
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8934012576937675
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.885959103703499
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8920385017991066
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8569916039705276
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8497369289398193
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.829560726881027
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.802697628736496
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8674124255776405
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.84306588023901
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9117077961564064
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9125575125217438
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9092776626348495
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8411544635891914
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.841131307184696
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.898265317082405
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.84572571516037
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9138024300336838
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.902901604771614
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.8960224092006683
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.9000038355588913
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.888073079288006
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.885702520608902
datasetREDDIT-BINARY,batch64,seed0,mode0.5,loss3.894919253885746
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss4.075496024723295
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.941114163096947
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.915579391431205
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.894605689410922
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8792686251145376
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.864867346196235
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8667290587968464
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.85680466060397
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.867431224146976
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.87359487557713
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8590279938299443
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8481136666068547
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8413130437271503
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.835138028181052
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8321086485174636
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.824670957613595
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8218717363816275
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.816902747637109
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.812029325509373
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8090465098996704
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8057049754299697
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8048028915743286
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8020323650746404
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.797906925406637
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.8058406190027165
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7978408940230746
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.79652313642864
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.79581098315082
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7920056294791307
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7904167461998854
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7873999106733103
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7921318494820895
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.786413334592988
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7887304810029043
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.785191886032684
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7835755770719506
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.788741422604911
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7854190174537368
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7909026990962933
datasetREDDIT-MULTI-5K,batch64,seed0,mode0.5,loss3.7856833270833463
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss4.257217928767204
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss4.121266335248947
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss4.045133724808693
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.984748885035515
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.9508071839809418
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.939188852906227
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.9517511278390884
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss4.00215108692646
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.953932225704193
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.9256687611341476
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.9092568904161453
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.9115889072418213
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.90236596763134
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.9081001579761505
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8866583853960037
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8884065747261047
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8838792741298676
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8787295669317245
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8780657351017
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8764145970344543
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8841723799705505
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8692695051431656
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.874669671058655
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8647124618291855
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.864464983344078
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8703732788562775
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8584358990192413
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.863645985722542
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8556124567985535
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8674887865781784
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8621288537979126
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8514245450496674
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.860722988843918
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.864264115691185
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.842862904071808
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8608859479427338
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.852433353662491
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.852279856801033
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.8527045100927353
datasetIMDB-BINARY,batch64,seed0,mode0.5,loss3.840368375182152
