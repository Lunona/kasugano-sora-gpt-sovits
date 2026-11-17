import argparse
import copy
import random
import time

from performance import (
    ROOT,S1,CHANNELS,EMOTIONS,annotated_plan,features,install,
    load_annotations,torch,
)
from AR.models.t2s_lightning_module import Text2SemanticLightningModule

HOLDOUT={"scenario/00_a030.ks","scenario/00_c010b.ks"}

def load_model(device):
    base=torch.load(S1,map_location="cpu",weights_only=False)
    config=copy.deepcopy(base["config"])
    model=Text2SemanticLightningModule(config,ROOT,is_train=False)
    model.load_state_dict(base["weight"])
    return model.model.to(device),config

def loader():
    annotation=load_annotations()
    path=ROOT/"prep/6-name2semantic-0.tsv"
    items={}
    items_test={}
    for line in path.read_text(encoding="utf-8").splitlines():
        name,y=line.split("\t")
        voice=name.removesuffix(".wav")
        row=annotation[voice]
        text=row["text_ja"]
        if not text or not y or not row["initial_training_candidate"]:
            continue
        plan=annotated_plan(row)
        ph,cond=features(text,plan)
        labels=torch.tensor([int(s) for s in y.split()],dtype=torch.long)
        if not ph or not labels.numel():
            continue
        item=(torch.tensor(ph),labels,cond)
        target=items_test if row["script"] in HOLDOUT else items
        target[voice]=item
    assert len(items)>2000 and items_test
    return items,items_test

def batch(items, device, conditioned=True):
    n=len(items)
    phone_max=max(len(x) for x,_,_ in items)
    code_max=max(len(y) for _,y,_ in items)
    phones=torch.zeros(n,phone_max,dtype=torch.long)
    codes=torch.zeros(n,code_max,dtype=torch.long)
    cond=torch.zeros(n,1024,phone_max)
    xl,yl=[],[]
    for i,(x,y,c) in enumerate(items):
        phones[i,-len(x):]=x
        codes[i,-len(y):]=y
        if conditioned:
            cond[i,:CHANNELS,-len(x):]=c
        xl.append(len(x))
        yl.append(len(y))
    return [t.to(device) for t in (phones,torch.tensor(xl),codes,torch.tensor(yl),cond)]

def loss_for(decoder, args):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        loss,_=decoder.forward_old(*args)
        return loss/args[3].sum()

def evaluate(model, test, device):
    with torch.no_grad():
        actual=[]; control=[]
        for i in range(0,len(test),4):
            subset=test[i:i+4]
            actual.append(float(loss_for(model,batch(subset,device))))
            control.append(float(loss_for(model,batch(subset,device,False))))
        return sum(actual)/len(actual),sum(control)/len(control)

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--epochs",type=int,default=2)
    opt=arg.parse_args()
    random.seed(72)
    torch.manual_seed(72)
    device="cuda"
    assert torch.cuda.is_available()
    items,test=loader()
    decoder,_=load_model(device)
    adapter=install(decoder)
    decoder.eval()
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.001)
    results=ROOT/"results"
    results.mkdir(exist_ok=True)
    evaluation=list(test.values())[:64]
    print(f"Train {len(items)}, held-out {len(test)}, evaluation {len(evaluation)}; pretrain",evaluate(decoder,evaluation,device),flush=True)
    usable=sum((condition[:-2].sum(dim=0)>0).any().item() for _,_,condition in items.values())
    shifts=sum(len(torch.unique(condition[:len(EMOTIONS)].T,dim=0))>1 for _,_,condition in items.values())
    print(f"Training items with emotion conditioning: {usable}, changing primary channels: {shifts}",flush=True)
    ordered=list(items.values())
    counter=0
    for epoch in range(opt.epochs):
        random.shuffle(ordered)
        for start in range(0,len(ordered),80):
            block=sorted(ordered[start:start+80],key=lambda item:len(item[1]))
            for i in range(0,len(block),4):
                args=batch(block[i:i+4],device)
                optimizer.zero_grad(set_to_none=True)
                loss=loss_for(decoder,args)
                assert loss.isfinite()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
                optimizer.step()
                counter+=1
                if counter%100==0:
                    print(epoch+1,counter,float(loss.detach()),flush=True)
        actual,control=evaluate(decoder,evaluation,device)
        saved={"adapter":adapter.state_dict(),"emotions":EMOTIONS,"steps":counter,
               "train_items":len(items),"holdout":list(HOLDOUT),"loss":(actual,control)}
        checkpoint=results/f"phrase-adapter-epoch{epoch+1}.pt"
        tmp=checkpoint.with_suffix(".pending")
        torch.save(saved,tmp)
        tmp.replace(checkpoint)
        print("End",epoch+1,actual,control,checkpoint,flush=True)

if __name__=="__main__":
    main()
