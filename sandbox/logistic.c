//doug has made code for logistic regression (including spa)
//compile this using gcc -lm logistic.c -o logistic.out
//run using ./logistic.out data.txt num_samples num_preds num_covars
//which data.txt has num_samples rows and 1+num_preds+num_covars columns

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

extern void dgemm_();
extern void dgemv_();
extern void dsyev_();

////////

//this function performs logistic spa
int spa_logistic(double score, double *data, int ns, double *probs, double *weights, double *stats)
{
int i, count, flag;
double value, value2, value3, diff, relax;

double kk, kkb, k0, k1, k1b, k2, ww, vv, ss, ssold;


//K0(t) is sum ( log (probs exp(tX) + (1-probs)) ) - t XT probs
//K1(t) is sum ( X probs / (probs + (1-probs) exp(-tX)) ) - XT probs
//K2(t) is sum ( X^2 probs (1-probs) exp(-tX) / (probs + (1-probs) exp(-tX))^2 )

//useful to compute sum (data * probs)
value3=0;for(i=0;i<ns;i++){value3+=data[i]*probs[i];}

//flag will indicate success: 1=good, 2=approx (timeout), -1=fail

//starting knot is zero
kk=0;

//work out starting cgfs (k0 and k1 are trivial)
k0=0;k1=0;
k2=0;
for(i=0;i<ns;i++){k2+=pow(data[i],2)*weights[i];}

//starting stat is non-spa stat
ssold=score*pow(k2,-.5);

count=0;
while(1)
{
if(count==10)
{
printf("Warning, SPA did not converge within %d iterations\n\n", count);
if(flag==1){flag=2;}
break;
}

//work out full move
diff=(score-k1)/k2;

flag=-1;
relax=1;
while(relax>0.001)
{
//get proposed move and corresponding k1
kkb=kk+relax*diff;
k1b=-value3;
for(i=0;i<ns;i++)
{k1b+=data[i]*probs[i]/(probs[i]+(1-probs[i])*exp(-data[i]*kkb));}

if(fabs(k1b-score)<fabs(k1-score))	//good move - update knot and cgfs and break
{
kk=kkb;
k1=k1b;
k0=-kk*value3;k2=0;
for(i=0;i<ns;i++)
{
value=exp(-data[i]*kk);
value2=probs[i]+(1-probs[i])*value;
k0+=log(value2/value);
k2+=pow(data[i],2)*weights[i]*value*pow(value2,-2);
flag=1;
}
break;
}
else	//bad move - will try smaller move
{relax*=0.5;}
}

if(flag==-1)	//failed to move
{break;}

if(kk*score-k0>0&&k2>0)	//can compute the spa statistic and maybe break
{
ww=pow(2*(kk*score-k0),.5);
if(kk<0){ww=-ww;}
vv=kk*pow(k2,.5);
ss=(ww+pow(ww,-1)*log(vv/ww));

if(fabs(ss-ssold)<0.01){break;}

ssold=ss;
}
else	//failed this time - hopefully will work later
{flag=-1;}

count++;
}

if(flag==1||flag==2)	//load up stats
{
//update test statistic
stats[2]=ss;

//set SE so consistent with effect size (stats[0]) and test statistic
stats[1]=stats[0]/stats[2];

//update p-value
stats[3]=erfc(fabs(stats[2])*M_SQRT1_2);
}

return(flag);
}

////////

//this function simply inverts the matrix mat
void invert_matrix(double *mat, int length)
{
int i, j, count, count2, lwork, info;
double wkopt, value, alpha, beta;
double *mat2, *mat3, *work;

mat2=malloc(sizeof(double)*length);
mat3=malloc(sizeof(double)*length*length);

//this works out required size of work vector
lwork=-1;
dsyev_("V", "U", &length, mat, &length, mat2, &wkopt, &lwork, &info);
if(info!=0)
{printf("Error, eigen priming failed; please tell Doug (info %d, length %d)\n\n", info, length);exit(1);}
lwork=(int)wkopt;
work=malloc(sizeof(double)*lwork);

//this does eigen decomposition
dsyev_("V", "U", &length, mat, &length, mat2, work, &lwork, &info);
if(info!=0)
{printf("Error, eigen decomp failed; please tell Doug (info %d, length %d)\n\n", info, length);exit(1);}
free(work);

//get number of (strongly) negative eigenvalues
count=0;for(i=0;i<length;i++){count+=(mat2[i]<=-0.000001);}

//and number of positive eigenvalues
count2=length-count;

//load U|E|^⁻.5 into mat3
#pragma omp parallel for private(j,i,value) schedule(static)
for(j=0;j<length;j++)
{
if(fabs(mat2[j])>0.000001)
{
value=pow(fabs(mat2[j]),-.5);
for(i=0;i<length;i++){mat3[j*length+i]=mat[j*length+i]*value;}
}
else
{
for(i=0;i<length;i++){mat3[(size_t)j*length+i]=0;}
}
}

if(count2>0)	//deal with last count2 vectors (positive eigenvalues)
{
alpha=1.0;beta=0.0;
dgemm_("N", "T", &length, &length, &count2, &alpha, mat3+count*length, &length, mat3+count*length, &length, &beta, mat, &length);
}

if(count>0)	//deal with first count vectors (negative eigenvalues)
{
if(count2>0){alpha=-1.0;beta=1.0;}
else{alpha=-1.0;beta=0.0;}
dgemm_("N", "T", &length, &length, &count, &alpha, mat3, &length, mat3, &length, &beta, mat, &length);
}

free(mat2);free(mat3);
}

////////

//this function replaces X with residual from regressing X on Z with weights
void reg_covar_weighted(double *X, double *Z, int ns, int length, int nf, double *weights)
{
int i, j;
double alpha, beta;
double *WZ, *ZTWZ, *ZTWX, *thetas;


WZ=malloc(sizeof(double)*ns*nf);
ZTWZ=malloc(sizeof(double)*nf*nf);
ZTWX=malloc(sizeof(double)*nf*length);
thetas=malloc(sizeof(double)*nf*length);

//set WZ
#pragma omp parallel for private(j, i) schedule(static)
for(j=0;j<nf;j++)
{
for(i=0;i<ns;i++){WZ[i+j*ns]=Z[i+j*ns]*weights[i];}
}

//compute ZTWZ and ZTWX
alpha=1.0;beta=0.0;
dgemm_("T", "N", &nf, &nf, &ns, &alpha, WZ, &ns, Z, &ns, &beta, ZTWZ, &nf);
dgemm_("T", "N", &nf, &length, &ns, &alpha, WZ, &ns, X, &ns, &beta, ZTWX, &nf);

//get inverse of ZTWZ
invert_matrix(ZTWZ, nf);

//thetas are inv(ZTWZ) ZTWX
alpha=1.0;beta=0.0;
dgemm_("N", "N", &nf, &length, &nf, &alpha, ZTWZ, &nf, ZTWX, &nf, &beta, thetas, &nf);

//X becomes X - Z thetas
alpha=-1.0;beta=1.0;
dgemm_("N", "N", &ns, &length, &nf, &alpha, Z, &ns, thetas, &nf, &beta, X, &ns);

free(WZ);free(ZTWZ);free(ZTWX);free(thetas);
}

////////

//this matrix estimates coefficients for the null model
void log_reg_null(double *Y, double *Z, int ns, int nf, double *nullprobs, double *nullweights, int maxiter, double tol)
{
int i, j, count, one=1;
double mean, sum, relax, like, like2, likeold, alpha, beta;
double *thetas, *thetadiffs, *probs, *probs2, *Zthetas, *Z2, *AI, *BI;

thetas=malloc(sizeof(double)*nf);
thetadiffs=malloc(sizeof(double)*nf);
probs=malloc(sizeof(double)*ns);
probs2=malloc(sizeof(double)*ns);
Zthetas=malloc(sizeof(double)*ns);
Z2=malloc(sizeof(double)*ns*nf);
AI=malloc(sizeof(double)*nf*nf);
BI=malloc(sizeof(double)*nf);

//get mean phenotype
sum=0;for(i=0;i<ns;i++){sum+=Y[i];}
mean=sum/ns;

if(mean==0||mean==1)	//trivial
{printf("Error, the phenotype is trivial (either all zeros or all ones)\n\n");}

//first effect starts at log(mean)-log(1-mean), rest at zero
thetas[0]=log(mean)-log(1-mean);
for(j=1;j<nf;j++){thetas[j]=0;}

//starting probs are mean
for(i=0;i<ns;i++){probs[i]=mean;}

//get starting likelihood
like=0;for(i=0;i<ns;i++){like+=Y[i]*log(probs[i])+(1-Y[i])*log(1-probs[i]);}

count=0;
relax=1;
while(1)	//iterate until likelihood converges
{
if(count==maxiter){printf("Error, did not converge within %d iterations\n\n", count);exit(1);}

//convenient to get Z2=Z*probs*(1-probs)
#pragma omp parallel for private(j, i) schedule(static)
for(j=0;j<nf;j++)
{
for(i=0;i<ns;i++){Z2[i+j*ns]=Z[i+j*ns]*probs[i]*(1-probs[i]);}
}

//first derivative vector is t(Z) (Y-probs)
#pragma omp parallel for private(j, i) schedule(static)
for(j=0;j<nf;j++)
{
BI[j]=0;for(i=0;i<ns;i++){BI[j]+=(Y[i]-probs[i])*Z[i+j*ns];}
}

//minus second derivative matrix is t(Z) probs (1-probs) Z
alpha=1.0;beta=0.0;
dgemm_("T", "N", &nf, &nf, &ns, &alpha, Z, &ns, Z2, &ns, &beta, AI, &nf);

//invert AI
(void) invert_matrix(AI, nf);

//proposed moves are inv AI BI
alpha=1.0;beta=0.0;
dgemv_("N", &nf, &nf, &alpha, AI, &nf, BI, &one, &beta, thetadiffs, &one);

//try to move
likeold=like;
relax=1;
while(relax>0.0001)
{
//propose moving relax*thetadiffs
for(j=0;j<nf;j++){thetas[j]+=relax*thetadiffs[j];}

//get corresponding probabilities
alpha=1.0;beta=0.0;
dgemv_("N", &ns, &nf, &alpha, Z, &ns, thetas, &one, &beta, Zthetas, &one);
for(i=0;i<ns;i++){probs2[i]=pow(1+exp(-Zthetas[i]),-1);}

//get likelihood of proposed move
like2=0;for(i=0;i<ns;i++){like2+=Y[i]*log(probs2[i])+(1-Y[i])*log(1-probs2[i]);}

if(like2>like-tol)	//accept move
{
like=like2;
for(i=0;i<ns;i++){probs[i]=probs2[i];}
break;
}
else	//move back and next turn try smaller move
{
for(j=0;j<nf;j++){thetas[j]-=relax*thetadiffs[j];}
relax*=.5;
}
}

//see if breaking
if(fabs(like-likeold)<tol){break;}

count++;
}

//load up nullprobs and nullweights
for(i=0;i<ns;i++){nullprobs[i]=probs[i];nullweights[i]=probs[i]*(1-probs[i]);}

free(thetas);free(thetadiffs);free(probs);free(probs2);free(Zthetas);free(Z2);free(AI);free(BI);
}

////////


int main (int argc, const char * argv[])
{
int i, j, one=1, info;
int num_samples, num_preds, num_covars, num_fixed;
char datafile[500];

double *X, *Z, *Y;

double value, value2, alpha, beta;
double *nullprobs, *nullweights, *Yadj, *XTYadj, *XTWX, *stats;


char readchar;
FILE *input, *output;

//read arguments

if(argc!=5)
{printf("Error, there must be four arguments (not %d), providing data file, sample size, number of covariates and number of predictors\n\n", argc-1);exit(1);}

strcpy(datafile,argv[1]);

if(sscanf(argv[2],"%d%c", &num_samples, &readchar)!=1)
{printf("Error, the second argument should be an integer (not %s)\n\n", argv[2]);exit(1);}

if(sscanf(argv[3],"%d%c", &num_covars, &readchar)!=1)
{printf("Error, the third argument should be an integer (not %s)\n\n", argv[3]);exit(1);}

if(sscanf(argv[4],"%d%c", &num_preds, &readchar)!=1)
{printf("Error, the four argument should be an integer (not %s)\n\n", argv[4]);exit(1);}

printf("Will read data for %d samples, %d predictors and %d covariates from %s\n\n", num_samples, num_preds, num_covars, datafile);

////////

//will add an intercept
num_fixed=1+num_covars;

////////

//load up data
Y=malloc(sizeof(double)*num_samples);
Z=malloc(sizeof(double)*num_samples*num_fixed);
X=malloc(sizeof(double)*num_samples*num_preds);

if((input=fopen(datafile,"r"))==NULL)
{printf("Error opening %s\n\n",datafile);exit(1);}

for(i=0;i<num_samples;i++)
{
if(i%100==0){printf("Reading Row %d of %d\n", i+1, num_samples);}

//read Y
if(fscanf(input, "%lf ", Y+i)!=1)
{printf("Error reading first element of Row %d of %s\n\n", i+1, datafile);exit(1);}

if(Y[i]!=0&&Y[i]!=1)
{printf("Error, the first element of Row %d of %s is %f (not zero or one)\n\n", i+1, datafile, Y[i]);exit(1);}

//first covariate is one
Z[i]=1;

//read remaining covariates
for(j=1;j<num_fixed;j++)
{
if(fscanf(input, "%lf ", Z+i+j*num_samples)!=1)
{printf("Error reading Element %d of Row %d of %s\n\n", 2+j, i+1, datafile);exit(1);}
}

//read predictors
for(j=0;j<num_preds;j++)
{
if(fscanf(input, "%lf ", X+i+j*num_samples)!=1)
{printf("Error reading Element %d of Row %d of %s\n\n", 1+num_covars+j+1, i+1, datafile);exit(1);}
}
}

fclose(input);

for(i=0;i<num_samples;i++)
{
if(i<5)
{
printf("Ind %d has phenotype %.2f, covariates", i+1, Y[i]);
for(j=0;j<num_fixed;j++){printf(" %.2f", Z[i+j*num_samples]);}
printf(", and predictors");
for(j=0;j<num_preds;j++){printf(" %.2f", X[i+j*num_samples]);}
printf("\n");
}
}
printf("\n");

////////

//solve the null model (just regress Y on Z)
nullprobs=malloc(sizeof(double)*num_samples);
nullweights=malloc(sizeof(double)*num_samples);
log_reg_null(Y, Z, num_samples, num_fixed, nullprobs, nullweights, 100, 1e-6);

//the adjusted response is (Y-nullprobs)
Yadj=malloc(sizeof(double)*num_samples);
for(i=0;i<num_samples;i++){Yadj[i]=Y[i]-nullprobs[i];}

//note that all above needs only to be performed once (not once per variant tested)

////////

//regress covariates out of X
reg_covar_weighted(X, Z, num_samples, num_preds, num_fixed, nullweights);

//compute t(X) Yadj
XTYadj=malloc(sizeof(double)*num_preds);

alpha=1.0;beta=0.0;
dgemv_("T", &num_samples, &num_fixed, &alpha, X, &num_samples, Yadj, &one, &beta, XTYadj, &one);

//compute t(X) W X
XTWX=malloc(sizeof(double)*num_preds);

#pragma omp parallel for private(j,i) schedule(static)
for(j=0;j<num_preds;j++)
{
XTWX[j]=0;for(i=0;i<num_samples;i++){XTWX[j]+=pow(X[i+j*num_samples],2)*nullweights[i];}
}

//can now test predictors - will compute effect, SE, test stat, p-value
stats=malloc(sizeof(double)*num_samples*4);

for(j=0;j<num_preds;j++)
{
//the score statistic is YTdata, with variance XTWX
//SAIGE paper explains how score/variance approx equals LogOR, and 1/var approx equals Var of LogOR
//this is because XT(Y-mu)/XTWX is 1-step NR estimate of LogOR
stats[0+j*4]=XTYadj[j]/XTWX[j];
stats[1+j*4]=pow(XTWX[j],-.5);
stats[2+j*4]=stats[0+j*4]/stats[1+j*4];
stats[3+j*4]=erfc(fabs(stats[2+j*4])*M_SQRT1_2);

printf("Predictor %d, test statistic %f, pvalue %e\n", j+1, stats[2+j*4], stats[3+j*4]);
}
printf("\n");

//do spa for predictors with p-value < 0.05
for(j=0;j<num_preds;j++)
{
if(stats[3+j*4]<0.05)
{
printf("Retest predictor %d, test statistic %f, pvalue %e\n", j+1, stats[2+j*4], stats[3+j*4]);

info=spa_logistic(XTYadj[j], X+j*num_samples, num_samples, nullprobs, nullweights, stats+j*4);

printf("SPA test statistic %f, pvalue %e\n", stats[2+j*4], stats[3+j*4]);
}
}

free(Y);free(Z);free(X);
free(nullprobs);free(nullweights);free(Yadj);
free(XTYadj);free(XTWX);
free(stats);

return(1);
}



