import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np, pickle, zipfile, io, collections, json, time, csv, hashlib
from pathlib import Path
OLD=Path(r'C:\Users\Edwin\Documents\Codex\2026-10-22\i-am-working-on-huawei-tech-3')
W=Path(__file__).resolve().parent; O=W.parent/'outputs'
def weights(path):
    # Restricted reader for the numeric tensors in these local PyTorch state dictionaries.
    with zipfile.ZipFile(path) as z:
        prefix=z.namelist()[0].split('/')[0]+'/'
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
                if (module,name)==('torch','FloatStorage'):return np.dtype('<f4')
                if (module,name)==('torch._utils','_rebuild_tensor_v2'):
                    return lambda a,off,shape,stride,*args: np.ndarray(shape,dtype=a.dtype,buffer=a,offset=off*a.itemsize,strides=tuple(s*a.itemsize for s in stride)).copy()
                raise ValueError((module,name))
            def persistent_load(self,pid):
                assert pid[0]=='storage' and pid[3]=='cpu'
                return np.frombuffer(z.read(prefix+'data/'+pid[2]),dtype=pid[1])
        return Reader(io.BytesIO(z.read(prefix+'data.pkl'))).load()
def auc(y,p):
    y=np.asarray(y).ravel();p=np.asarray(p).ravel();order=np.argsort(p,kind='stable');s=p[order];a=np.r_[0,np.flatnonzero(np.diff(s))+1];b=np.r_[a[1:],len(s)];r=np.repeat((a+b+1)/2,b-a);n=y.sum()
    return float((r@y[order]-n*(n+1)/2)/(n*(len(y)-n))) if 0<n<len(y) else None
def first(d,cut):
    ix=np.flatnonzero(d['t']>=cut);order=ix[np.argsort(d['t'][ix],kind='stable')];_,a=np.unique(d['u'][order],return_index=True);return np.sort(order[a])
def prob(z):
    a=np.exp(np.clip(z,-50,40));return a/(1+a.sum(1,keepdims=True))
def score(m,d,ix,du=None):
    out=[]
    for q in np.array_split(ix,max(1,(len(ix)+4095)//4096)):
        u=d['u'][q];it=d['i'][q];c=d['tb'][q].astype(int)+4*d['dt'][q]
        v=m['pref.u.weight'][u].copy()
        if du is not None:v+=du[u,:-1]
        z=np.einsum('bd,bkd->bk',v,m['pref.i.weight'][it])+m['pref.b.weight'][it,0]+m['ub.weight'][u]+m['bias']+m['ib.weight'][it,c[:,None]]+m['ut.weight'][u,c][:,None]
        if du is not None:z+=du[u,-1,None]
        out.append(prob(z))
    return np.concatenate(out)
def corrected(p,d,ix,raw,cut,ni):
    n=np.bincount(raw['i'][raw['t']<cut].ravel(),minlength=ni);ratio=(n[d['i'][ix]]+20)/(n.mean()+20);a=p/np.sqrt(ratio);return a/(np.maximum(1-p.sum(1,keepdims=True),1e-8)+a.sum(1,keepdims=True))
def fit_map(m,d,target,ix,cut,steps=30):
    start=time.time();nu,dim=m['pref.u.weight'].shape;ni=len(m['pref.i.weight']);D=dim+1
    # Summed candidate likelihood and first-snapshot warmup click likelihood.
    tr=np.flatnonzero(d['t']<cut);_,a=np.unique(d['u'][tr],return_index=True);fq=tr[a]
    hu=d['u'][fq];hh=d['h'][fq];ok=hh>=0
    sums=np.zeros((nu,dim));counts=np.zeros(nu)
    np.add.at(sums,hu,(m['pref.i.weight'][hh.clip(min=0)]*ok[:,:,None]).sum(1));np.add.at(counts,hu,ok.sum(1))
    # Use available histories for genuinely unseen users, without validation labels.
    seen=np.bincount(d['u'][tr],minlength=nu)>0;q=ix[~seen[target['u'][ix]]];u=target['u'][q];h=target['h'][q];ok=h>=0
    sums[u]=(m['pref.i.weight'][h.clip(min=0)]*ok[:,:,None]).sum(1);counts[u]=ok.sum(1)
    # Fixed Monte Carlo catalogue normalization, independent of validation labels.
    rng=np.random.default_rng(104);neg=rng.integers(ni,size=(nu,128));nv=m['pref.i.weight'][neg];nb=m['pref.b.weight'][neg,0]
    du=np.zeros((nu,D));anchor=m['pref.u.weight'].astype(float).copy();du[~seen,:dim]=-anchor[~seen]
    mom=np.zeros_like(du);var=np.zeros_like(du);eventn=np.bincount(d['u'][tr],minlength=nu)
    for step in range(1,steps+1):
        g=2*du;g[~seen,:dim]=2*(anchor[~seen]+du[~seen,:dim]);g[:,-1]*=4
        for qs in np.array_split(np.arange(nu),14):
            z=np.einsum('ud,ukd->uk',anchor[qs]+du[qs,:dim],nv[qs])+nb[qs];z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
            g[qs,:dim]+=counts[qs,None]*np.einsum('uk,ukd->ud',p,nv[qs])-sums[qs]
        for q in np.array_split(tr,max(1,(len(tr)+8191)//8192)):
            u=d['u'][q];it=d['i'][q];c=d['tb'][q].astype(int)+4*d['dt'][q];v=m['pref.i.weight'][it]
            z=np.einsum('bd,bkd->bk',anchor[u]+du[u,:dim],v)+m['pref.b.weight'][it,0]+m['ub.weight'][u]+m['bias']+m['ib.weight'][it,c[:,None]]+m['ut.weight'][u,c][:,None]+du[u,-1,None]
            e=prob(z)-d['y'][q];np.add.at(g[:,:dim],u,np.einsum('bk,bkd->bd',e,v));np.add.at(g[:,-1],u,e.sum(1))
        g/=np.maximum(eventn+counts+2,1)[:,None]
        mom=.9*mom+.1*g;var=.999*var+.001*g*g;du-=.06*(mom/(1-.9**step))/(np.sqrt(var/(1-.999**step))+1e-8)
        if step%10==0:print('MAP step',step,'seconds',round(time.time()-start,1),flush=True)
    return du,round(time.time()-start,2)
def slices(raw,ix,p,cut):
    h=(raw['h'][ix]>=0).sum(1);un=np.bincount(raw['u'][raw['t']<cut],minlength=int(raw['u'].max())+1);inn=np.bincount(raw['i'][raw['t']<cut].ravel(),minlength=50000);repeat=(raw['i'][ix,:,None]==raw['h'][ix,None,:]).any(2)
    masks={f'history_{a}_{b}':np.broadcast_to(((h>=a)&(h<=b))[:,None],p.shape) for a,b in [(0,0),(1,5),(6,10),(11,19),(20,20)]}
    masks.update(unseen_user=np.broadcast_to((un[raw['u'][ix]]==0)[:,None],p.shape),unseen_item=inn[raw['i'][ix]]==0,repeat=repeat,nonrepeat=~repeat)
    for t in range(4):masks['time_bucket_'+str(t)]=np.broadcast_to((raw['tb'][ix]==t)[:,None],p.shape)
    for a,b in [(0,2),(3,10),(11,10000)]:masks[f'activity_{a}_{b}']=np.broadcast_to(((un[raw['u'][ix]]>=a)&(un[raw['u'][ix]]<=b))[:,None],p.shape)
    y=raw['y'][ix];return {k:dict(rows=int(v.sum()),clicks=int(y[v].sum()),auc=auc(y[v],p[v])) for k,v in masks.items()}
def main():
    raw=dict(np.load(OLD/'work/train.csv.npz'));cut=1737849600;ix=first(raw,cut);m=weights(OLD/'work/finetune_8_best.pt');p=score(m,raw,ix);saved=np.load(OLD/'work/finetune_8_validation.npz');assert np.array_equal(ix,saved['ix']);err=float(np.max(np.abs(p-saved['p'])));assert err<1e-6
    meta=np.load(OLD/'work/meta_validation.npz')['p'].reshape(-1,10);base=corrected(p,raw,ix,raw,cut,50000);blend=.5*(base+meta)
    report=dict(checkpoint_max_prediction_error=err,baseline_auc=auc(raw['y'][ix],blend),protocol='January 26 development only; January 24 reserved and not examined',slices_before=slices(raw,ix,blend,cut))
    print('BASELINE',report['baseline_auc'],'max_error',err,flush=True)
    du,seconds=fit_map(m,raw,raw,ix,cut);np.savez(W/'map_checkpoint.npz',delta=du)
    records=[]
    for scale in [.25,.5,1.]:
        pp=corrected(score(m,raw,ix,du*scale),raw,ix,raw,cut,50000);bb=.5*(meta+pp);rec=dict(scale=scale,preference_auc=auc(raw['y'][ix],pp),blend_auc=auc(raw['y'][ix],bb));records.append(rec);print('RESULT',rec,flush=True);np.savez(W/f'map_predictions_{scale}.npz',p=bb,ix=ix)
    best=max(records,key=lambda x:x['blend_auc']);report.update(experiments=records,seconds=seconds,best=best,slices_after=slices(raw,ix,np.load(W/f"map_predictions_{best['scale']}.npz")['p'],cut))
    (O/'audit_experiment.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
