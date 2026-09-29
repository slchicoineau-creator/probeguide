"""Per-token raw-intensity features (baseline that uses no DINO): normalised CT stats in the token and its 3x3x3
token neighbourhood. Saved as emb/int_XXX.npy (32,32,32,F)."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, sys, numpy as np
from scipy.ndimage import uniform_filter
from probeguide.zio import RemoteArray, CT_2UM
ct=RemoteArray(CT_2UM,0)
for f in sorted(glob.glob("emb/chunk_*.npz")):
    out=f.replace("chunk_","int_").replace(".npz",".npy")
    d=np.load(f); x0,y0,z0=d["origin_xyz"]
    v=ct.read(z0,z0+256,y0,y0+256,x0,x0+256).astype(np.float32); lo,hi=np.percentile(v,[1,99]); v=np.clip((v-lo)/max(hi-lo,1e-6),0,1)
    t=v.reshape(32,8,32,8,32,8).transpose(0,2,4,1,3,5).reshape(32,32,32,512)
    feats=[t.mean(-1),t.std(-1),*np.percentile(t,[10,50,90],axis=-1),t.max(-1)]
    F=np.stack(feats,-1)
    nb=np.stack([uniform_filter(F[...,k],size=3,mode="nearest") for k in range(F.shape[-1])],-1)
    np.save(out,np.concatenate([F,nb],-1).astype(np.float32))
print("done")
