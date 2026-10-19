from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False,
                     'figure.dpi':150, 'axes.titleweight':'bold'})
(ROOT/'figures').mkdir(exist_ok=True)

import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
rng=np.random.default_rng(5262)
X=rng.normal(size=(450,5))
y=2*np.sin(X[:,0])+X[:,1]**2-.6*X[:,2]*X[:,3]+rng.normal(scale=.3,size=len(X))
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=5262)
models={'Ridge':Ridge(),'Polynomial kernel':KernelRidge(kernel='polynomial',degree=2),'RBF kernel':KernelRidge(kernel='rbf',gamma=.15)}
rows=[]
for name,model in models.items():
    pipe=Pipeline([('scale',StandardScaler()),('model',model)])
    search=GridSearchCV(pipe,{'model__alpha':[.01,.1,1.,10.]},cv=KFold(3,shuffle=True,random_state=5262),scoring='neg_mean_squared_error')
    search.fit(x_train,y_train)
    rows.append({'model':name,'test_mse':float(mean_squared_error(y_test,search.predict(x_test))),'alpha':search.best_params_['model__alpha']})
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
ax.bar([r['model'] for r in rows],[r['test_mse'] for r in rows],color=['#718eaa','#d69b3a','#157a83'])
ax.set(ylabel='Held-out mean squared error',title='Seeded nonlinear regression | synthetic data')
fig.savefig(ROOT/'figures/regression-comparison.svg')
(ROOT/'figures/results.json').write_text(json.dumps({'seed':5262,'training_n':len(x_train),'test_n':len(x_test),'models':rows},indent=2))
print(json.dumps(rows))
