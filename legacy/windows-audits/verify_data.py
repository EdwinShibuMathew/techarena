from audit_experiment import *
def inspect_archive(name,cache,users,items,limit=None):
    start=time.time();events=rows=0;ids=[];last=None;group=[]
    def check(g,j):
        assert len(g)==10 and len({a['event_id'] for a in g})==1
        a=g[0]
        for b in g:
            assert all(a[k]==b[k] for k in ['user_id','timestamp','time_bucket','day_type','history_item_ids','history_timestamps'])
        assert cache['u'][j]==users[a['user_id']]
        assert np.array_equal(cache['i'][j],[items[b['item_id']] for b in g])
        for k,field in [('t','timestamp'),('tb','time_bucket'),('dt','day_type')]:assert cache[k][j]==int(a[field])
        h=[items[x] for x in json.loads(a['history_item_ids'])];ht=json.loads(a['history_timestamps']);assert np.array_equal(cache['h'][j],h+[-1]*(20-len(h)));assert np.array_equal(cache['ht'][j],ht+[0]*(20-len(ht)))
        if 'clicked' in a:assert np.array_equal(cache['y'][j],[int(b['clicked']) for b in g])
    with zipfile.ZipFile(Path(r'C:\Users\Edwin\Downloads')/(name+'.zip')) as z:
        members=z.namelist()
        with io.TextIOWrapper(z.open(name+'.csv'),encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                if group and row['event_id']!=last:
                    check(group,events);events+=1;group=[]
                    if limit is not None and events>=limit:break
                group.append(row);last=row['event_id'];ids.append(row['rowid']);rows+=1
            else:
                if group:check(group,events);events+=1
    return dict(archive_members=members,events_compared=events,rows_compared=rows,seconds=round(time.time()-start,2)),ids
def main():
    users,items=pickle.load(open(OLD/'work/mappings.pkl','rb'));d=dict(np.load(OLD/'work/train.csv.npz'));e=dict(np.load(OLD/'work/evaluation.csv.npz'));report={}
    for name,x in [('train',d),('evaluation',e)]:
        ok=x['h']>=0;assert x['i'].shape==(len(x['u']),10);assert x['h'].shape==(len(x['u']),20);assert np.all((x['y']==0)|(x['y']==1)) and np.all(x['y'].sum(1)<=1);assert np.all(x['ht'][ok]<np.broadcast_to(x['t'][:,None],ok.shape)[ok]);assert np.all(np.diff((x['h']>=0).astype(int),axis=1)<=0)
        assert np.all(x['tb']==(x['t']//21600)%4);assert np.all(x['dt']==((x['t']//86400+3)%7>=5));assert np.all((np.diff(x['ht'],axis=1)>=0)|(~ok[:,1:]))
        report[name]=dict(events=len(x['u']),min_timestamp=int(x['t'].min()),max_timestamp=int(x['t'].max()),empty_histories=int((ok.sum(1)==0).sum()),cache_invariants_pass=True)
    report['train_archive_sample'],_=inspect_archive('train',d,users,items,2000)
    report['evaluation_archive_full'],ids=inspect_archive('evaluation',e,users,items)
    expected=pickle.load(open(OLD/'work/eval_rowids.pkl','rb'));assert ids==expected and len(set(ids))==len(ids)
    with open(r'C:\Users\Edwin\Downloads\example_submission.csv',newline='',encoding='utf-8-sig') as f:example=[r['rowid'] for r in csv.DictReader(f)]
    assert len(example)==len(ids) and set(example)==set(ids)
    report['all_evaluation_ids_match_cache_and_example']=True
    (O/'data_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()

