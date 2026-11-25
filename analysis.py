"""BIT4131 Statistical Computing: 23 Python demonstrations. Run: python analysis.py"""
from pathlib import Path
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats, optimize, linalg
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
ROOT=Path(__file__).parent; FIG=ROOT/"figures"; DATA=ROOT/"data"
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
rng=np.random.default_rng(42); results={}
def save(name): plt.tight_layout(); plt.savefig(FIG/name,dpi=150); plt.close()
# 1 Descriptive statistics
x=pd.Series([52,55,57,58,60,60,62,65,68,70,72,75]); results["1"]={"mean":x.mean(),"median":x.median(),"mode":x.mode().tolist(),"variance":x.var(),"sd":x.std(),"quartiles":x.quantile([.25,.5,.75]).tolist()}
plt.boxplot(x); plt.title("Boxplot"); save("01_boxplot.png")
# 2 Probability / conditional probability: two dice
out=[(a,b) for a in range(1,7) for b in range(1,7)]; A=[o for o in out if sum(o)==8]; B=[o for o in out if o[0]==3]
sim=rng.integers(1,7,(100000,2)); results["2"]={"P_sum_8":len(A)/36,"P_sum_8_given_first_3":sum(o in B for o in A)/len(B),"simulated":float(np.mean(sim.sum(1)==8))}
# 3 Distribution fit
v=rng.normal(50,8,500); mu,sd=stats.norm.fit(v); xx=np.linspace(v.min(),v.max(),200); plt.hist(v,25,density=True,alpha=.6); plt.plot(xx,stats.norm.pdf(xx,mu,sd)); plt.title("Fitted normal curve"); save("03_distribution.png"); results["3"]={"mu":mu,"sd":sd,"P_le_60":stats.norm.cdf(60,mu,sd)}
# 4 CLT
pop=rng.normal(100,15,100000); means=np.array([rng.choice(pop,40).mean() for _ in range(2500)]); plt.hist(means,30,density=True); plt.title("CLT: sample means"); save("04_clt.png"); results["4"]={"mean_of_means":means.mean(),"empirical_SE":means.std(ddof=1),"theoretical_SE":15/np.sqrt(40)}
# 5 Confidence interval
v=np.array([18,20,21,22,23,24,25,26,28,30,31,34]); results["5"]={"mean":v.mean(),"95%_CI":stats.t.interval(.95,len(v)-1,loc=v.mean(),scale=stats.sem(v))}
# 6 Paired hypothesis test
before=np.array([72,75,70,68,74,71,69,73,76,70]); after=before+np.array([4,3,3,4,3,3,3,3,3,4]); test=stats.ttest_rel(after,before); results["6"]={"t":test.statistic,"p":test.pvalue}
# 7 Correlation / covariance
df=pd.DataFrame({"hours":[1,2,3,4,5,6,7,8],"score":[48,52,55,60,64,68,73,78],"attendance":[60,65,70,72,80,82,88,92]}); results["7"]={"correlation":df.corr().round(3).to_dict(),"covariance":df.cov().round(2).to_dict()}
plt.scatter(df.hours,df.score); plt.xlabel("Hours"); plt.ylabel("Score"); save("07_correlation.png")
# 8 OLS regression
reg=pd.DataFrame({"hours":[1,2,2,3,4,4,5,6,6,7,8,9],"score":[42,48,51,53,58,62,65,68,72,75,81,86]}); fit=smf.ols("score~hours",reg).fit(); reg["fitted"]=fit.fittedvalues; reg["residual"]=fit.resid; reg.to_csv(DATA/"regression.csv",index=False); results["8"]={"coefficients":fit.params.to_dict(),"p_values":fit.pvalues.to_dict(),"R_squared":fit.rsquared}
plt.scatter(reg.hours,reg.score); plt.plot(reg.hours,reg.fitted,color="red"); plt.xlabel("Study hours"); plt.ylabel("Score"); save("08_regression.png")
plt.scatter(reg.fitted,reg.residual); plt.axhline(0,ls="--"); plt.xlabel("Fitted"); plt.ylabel("Residual"); save("08_residuals.png")
# 9 Matrix algebra
A=np.array([[4.,2.],[1.,3.]]); b=np.array([10.,8.]); results["9"]={"addition":(A+np.eye(2)).tolist(),"multiplication":(A@A).tolist(),"transpose":A.T.tolist(),"determinant":float(linalg.det(A)),"inverse":linalg.inv(A).tolist(),"solution":linalg.solve(A,b).tolist()}
# 10 Eigen / PCA
X=rng.normal(size=(120,4)); X[:,1]=.7*X[:,0]+.4*X[:,1]; Xs=StandardScaler().fit_transform(X); pca=PCA().fit(Xs); scores=pca.transform(Xs); results["10"]={"eigenvalues":pca.explained_variance_.tolist(),"explained_variance_ratio":pca.explained_variance_ratio_.tolist()}
plt.plot(np.arange(1,5),np.cumsum(pca.explained_variance_ratio_),marker="o"); plt.xlabel("Components"); plt.ylabel("Cumulative variance"); save("10_scree.png")
plt.scatter(scores[:,0],scores[:,1]); plt.xlabel("PC1"); plt.ylabel("PC2"); save("10_pca_scores.png")
# 11 Optimization
f=lambda z:(z[0]-3)**2+2*(z[1]+1)**2+4; opt=optimize.minimize(f,[0.,0.]); results["11"]={"optimum":opt.x.tolist(),"objective":opt.fun,"success":bool(opt.success)}
g=np.linspace(-1,5,80); xx,yy=np.meshgrid(g,g); plt.contour(xx,yy,(xx-3)**2+2*(yy+1)**2+4); plt.scatter(*opt.x,marker="x"); save("11_optimization.png")
# 12 Monte Carlo pi
pts=rng.random((100000,2)); inside=(pts**2).sum(1)<=1; running=4*np.cumsum(inside)/np.arange(1,len(inside)+1); results["12"]={"pi_estimate":float(running[-1])}
plt.plot(np.arange(1,len(running)+1,100),running[::100]); plt.axhline(np.pi,ls="--"); plt.xlabel("Draws"); plt.ylabel("Pi estimate"); save("12_monte_carlo.png")
# 13 Time series
idx=pd.date_range("2023-01-01",periods=60,freq="MS"); t=np.arange(60); ts=pd.Series(100+.8*t+10*np.sin(2*np.pi*t/12)+rng.normal(0,3,60),index=idx)
dec=seasonal_decompose(ts,period=12); forecast=ExponentialSmoothing(ts,trend="add",seasonal="add",seasonal_periods=12).fit().forecast(6); results["13"]={"forecast":forecast.round(2).tolist()}
ts.plot(); dec.trend.plot(); save("13_timeseries.png")
# 14 Visualization gallery
v=rng.normal(50,10,300); fig,ax=plt.subplots(2,2,figsize=(8,6)); ax[0,0].hist(v); ax[0,1].boxplot(v); ax[1,0].scatter(v,2*v+rng.normal(0,10,300)); ax[1,1].plot(np.arange(30),pd.Series(v[:30]).rolling(4).mean()); save("14_gallery.png")
# 15 Transformations
v=np.array([10,12,15,18,20,25]).reshape(-1,1); z=StandardScaler().fit_transform(v); mm=MinMaxScaler().fit_transform(v); pd.DataFrame({"original":v.ravel(),"z":z.ravel(),"minmax":mm.ravel()}).to_csv(DATA/"transformed.csv",index=False); results["15"]={"z_mean":float(z.mean()),"min":float(mm.min()),"max":float(mm.max())}
plt.plot(v.ravel(),label="Original"); plt.plot(z.ravel(),label="Z"); plt.plot(mm.ravel(),label="Minmax"); plt.legend(); save("15_transform.png")
# 16 ANOVA and Tukey
groups=[rng.normal(m,5,20) for m in [70,75,82]]; fstat,p=stats.f_oneway(*groups); ad=pd.DataFrame({"score":np.concatenate(groups),"group":np.repeat(["A","B","C"],20)}); am=smf.ols("score~C(group)",ad).fit(); anova=sm.stats.anova_lm(am,typ=2); tukey=pairwise_tukeyhsd(ad.score,ad.group); results["16"]={"F":fstat,"p":p,"means":ad.groupby("group").score.mean().to_dict(),"anova":anova.to_dict()}
# 17 Chi-square
obs=np.array([[30,20,10],[15,25,30]]); chi,p,dof,exp=stats.chi2_contingency(obs); results["17"]={"observed":obs.tolist(),"expected":exp.tolist(),"chi2":chi,"df":dof,"p":p}
# 18 Logistic regression
h=rng.normal(5,2,400); pr=1/(1+np.exp(-(-2+.7*h))); y=rng.binomial(1,pr); X=h.reshape(-1,1); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); clf=LogisticRegression().fit(Xtr,ytr); pred=clf.predict(Xte); results["18"]={"coefficients":clf.coef_.ravel().tolist(),"odds_ratio":np.exp(clf.coef_).ravel().tolist(),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"accuracy":accuracy_score(yte,pred)}
# 19 Nonparametric
a=[12,15,14,10,18,16,13,11]; b=[17,19,16,18,20,15,21,17]; u,p=stats.mannwhitneyu(a,b); results["19"]={"Mann_Whitney_U":u,"p":p,"Welch_t":stats.ttest_ind(a,b,equal_var=False).statistic}
# 20 Bootstrap
v=np.array([14,16,17,18,19,20,22,25,28,30]); boot=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(5000)]); ci=np.quantile(boot,[.025,.975]); results["20"]={"mean":v.mean(),"bootstrap_SE":boot.std(ddof=1),"CI":ci.tolist()}
plt.hist(boot,30,density=True); plt.axvline(ci[0],ls="--"); plt.axvline(ci[1],ls="--"); save("20_bootstrap.png")
# 21 Multivariate analysis
X=rng.normal(size=(100,5)); X[:,1]=.8*X[:,0]+rng.normal(0,.5,100); Z=StandardScaler().fit_transform(X); pc=PCA(n_components=2).fit_transform(Z); results["21"]={"correlation":np.corrcoef(X,rowvar=False).round(2).tolist()}
plt.scatter(pc[:,0],pc[:,1]); plt.xlabel("PC1"); plt.ylabel("PC2"); save("21_multivariate.png")
# 22 Missing data and median imputation
md=pd.DataFrame({"age":[21,25,np.nan,32,28,35,np.nan,40],"income":[30,42,50,np.nan,47,52,61,65],"score":[65,np.nan,72,80,75,88,90,np.nan]}); miss=md.isna().sum(); imp=pd.DataFrame(SimpleImputer(strategy="median").fit_transform(md),columns=md.columns); md.to_csv(DATA/"missing_before.csv",index=False); imp.to_csv(DATA/"missing_after.csv",index=False); results["22"]={"missing":miss.to_dict(),"remaining":int(imp.isna().sum().sum())}
plt.bar(md.columns,miss); save("22_missingness.png")
# 23 Reproducibility
results["23"]={"random_seed":42,"data_note":"Synthetic teaching data","workflow":"script + data + figures + results + README + requirements"}
(ROOT/"results.json").write_text(json.dumps(results,indent=2,default=lambda o:float(o)))
print("Completed 23 demonstrations. See results.json, data/ and figures/.")
