from audit_experiment import *
def aggregate(ids,values,size):
    return np.stack([np.bincount(ids,weights=values[:,j],minlength=size) for j in range(values.shape[1])],1).astype(np.float32)
class Adam:
    def __init__(self,params,lr):self.params=params;self.lr=lr;self.m={k:np.zeros_like(v) for k,v in params.items()};self.v={k:np.zeros_like(v) for k,v in params.items()};self.step=0
    def update(self,grads):
        self.step+=1
        for k,g in grads.items():
            self.m[k]*=.9;self.m[k]+=.1*g;self.v[k]*=.999;self.v[k]+=.001*g*g
            self.params[k]-=self.lr*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)
def train(raw,cut,nu,ni):
    rng=np.random.default_rng(42);tr=np.flatnonzero(raw['t']<cut);_,a=np.unique(raw['u'][tr],return_index=True);fq=tr[a];h=raw['h'][fq];ok=h>=0;y=raw['y'][tr].astype(bool)
    u=np.r_[np.broadcast_to(raw['u'][fq,None],h.shape)[ok],np.broadcast_to(raw['u'][tr,None],y.shape)[y]];it=np.r_[h[ok],raw['i'][tr][y]]
    p={'u':rng.normal(0,.1,(nu,8)).astype('f'),'i':rng.normal(0,.1,(ni,8)).astype('f'),'b':np.zeros((ni,1),'f')};opt=Adam(p,.012);start=time.time()
    for ep in range(40):
        order=rng.permutation(len(u))
        for st in range(0,len(order),4096):
            q=order[st:st+4096];us=u[q];cand=np.column_stack((it[q],rng.integers(ni,size=(len(q),31))));U=p['u'][us];V=p['i'][cand];z=np.einsum('bd,bkd->bk',U,V)+p['b'][cand,0];z-=z.max(1,keepdims=True);e=np.exp(z);e/=e.sum(1,keepdims=True);e[:,0]-=1;e/=len(q)
            gu=np.einsum('bk,bkd->bd',e,V)+.04*U/len(q);gi=e[:,:,None]*U[:,None,:]+.04*V/(len(q)*32)
            opt.update({'u':aggregate(us,gu,nu),'i':aggregate(cand.ravel(),gi.reshape(-1,8),ni),'b':aggregate(cand.ravel(),e.reshape(-1,1),ni)})
        if (ep+1)%5==0:print('clean pretrain',ep+1,'seconds',round(time.time()-start,1),flush=True)
    anchor={k:v.copy() for k,v in p.items()};p.update(ub=np.zeros((nu,1),'f'),ib=np.zeros((ni,8),'f'),ut=np.zeros((nu,8),'f'),bias=np.array([-2.5],dtype='f'));opt=Adam(p,.0015)
    for ep in range(8):
        order=rng.permutation(tr)
        for st in range(0,len(order),2048):
            q=order[st:st+2048];us=raw['u'][q];cand=raw['i'][q];c=raw['tb'][q].astype(int)+4*raw['dt'][q];U=p['u'][us];V=p['i'][cand];n=len(q)
            z=np.einsum('bd,bkd->bk',U,V)+p['b'][cand,0]+p['ub'][us]+p['bias']+p['ib'][cand,c[:,None]]+p['ut'][us,c][:,None];e=(prob(z)-raw['y'][q])/n
            gu=np.einsum('bk,bkd->bd',e,V)+.2*(U-anchor['u'][us])/n;gi=e[:,:,None]*U[:,None,:]+.2*(V-anchor['i'][cand])/(n*10)
            gib=.1*p['ib'][cand]/(n*10);gib[np.arange(n)[:,None],np.arange(10)[None,:],c[:,None]]+=e
            gut=.1*p['ut'][us]/n;gut[np.arange(n),c]+=e.sum(1)
            opt.update({'u':aggregate(us,gu,nu),'i':aggregate(cand.ravel(),gi.reshape(-1,8),ni),'b':aggregate(cand.ravel(),e.reshape(-1,1),ni),'ub':aggregate(us,e.sum(1)[:,None]+.1*p['ub'][us]/n,nu),'ib':aggregate(cand.ravel(),gib.reshape(-1,8),ni),'ut':aggregate(us,gut,nu),'bias':np.array([e.sum()])})
        print('clean finetune',ep+1,'seconds',round(time.time()-start,1),flush=True)
    m={('pref.'+k+'.weight' if k in ['u','i','b'] else k+'.weight' if k!='bias' else k):v for k,v in p.items()};np.savez(W/'clean_jan24_model.npz',**m);return m,time.time()-start
if __name__=='__main__':
    raw=dict(np.load(OLD/'work/train.csv.npz'));cut=1737676800;users,items=pickle.load(open(OLD/'work/mappings.pkl','rb'))
    # Freeze configuration before observing held-out labels. Only first events within Jan 24.
    ix=first(raw,cut);ix=ix[raw['t'][ix]<cut+86400]
    m,seconds=train(raw,cut,len(users),len(items));base=corrected(score(m,raw,ix),raw,ix,raw,cut,len(items));du,secs=fit_map(m,raw,raw,ix,cut);new=corrected(score(m,raw,ix,.25*du),raw,ix,raw,cut,len(items))
    report=dict(cutoff='2025-01-24T00:00:00Z',end='2025-01-25T00:00:00Z',events=len(ix),baseline_auc=auc(raw['y'][ix],base),map_auc=auc(raw['y'][ix],new),train_seconds=seconds,map_seconds=secs,pretrain_epochs=40,finetune_epochs=8,scale=.25,limitations='Retrospective temporal check of preference adjustment, not of original CatBoost blend; hyperparameters inherited from later-date work. No January 24 score used for selection in this run. NumPy training differs in random stream from PyTorch.')
    np.savez(W/'clean_jan24_predictions.npz',ix=ix,y=raw['y'][ix],base=base,new=new);(O/'clean_fold_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
