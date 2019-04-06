 import numpy as np
 import pandas as pd
 import nltk
 import sys
 import sklearn
 import nltk.classify.util
 from nltk.classify import NaiveBayesClassifier
 from nltk import pos_tag,word_tokenize
 from sklearn.naive_bayes import GaussianNB
 from sklearn.ensemble import RandomForestClassifier
 from sklearn import svm
 from sklearn.metrics import f1_score
 from mlxtend.plotting import plot_decision_regions
# import kindle reviews dataset
 df=pd.read_csv("C://Users/PrishitaRay/Documents/DM_Project/kindle_reviews.csv")
 df=df.replace("",np.nan)
 df=df.dropna(how="any",axis=0)
 ar=df.values
 print(array[0])
 print(len(array))
 rev_data=array[:,3]
 print(rev_data[0])
 rating=array[:,2]
 print(rating[0])

#Sentiment Sentences extraction and POS (parts of speech) tagging
 L=[]
 for i in range(0,978780):
	l=[]
	l.append(pos_tag(word_tokenize(rev_data[i])))
	L.append(l)

 print(L[49999])

#Negative phrases identification
#List of negative prefixes
 neg_pref=['no','not','never','n\'t','cannot']
 noa=[]
 nov=[]
 neg=[]
 flag=0
 flag1=0
 s=""
 cnt=0
# verb tags with one of the negative prefixes before them are appended to nov list (negative of verbs)
# adjective tags with one of the negative prefixes before them are appended to noa list (negative of adjectives)
 for x in L:
	for y in x[0]:
		if flag==1:
			if y[1]=='VB' or y[1]=='VBD' or y[1]=='VBG' or y[1]=='VBN' or y[1]=='VBP' or y[1]=='VBZ':
				st=s+" "+y[0]
				for t in nov:
					if t==st:
						flag1=1
				if flag1==0:
					if len(st.split())<=3:
						nov.append(st)
						neg.append(st)
				else:
					flag1=0
				flag=0
			elif y[1]=='JJ' or y[1]=='JJR' or y[1]=='JJS':
				st=s+" "+y[0]
				for t in noa:
					if t==st:
						flag1=1
				if flag1==0:
					if len(st.split())<=3:
						noa.append(st)
						neg.append(st)
				else:
					flag1=0
				flag=0
			else:
				s=s+" "+y[0]
				continue
		for z in neg_pref:
			if y[0]==z:
				flag=1
				s=y[0]
				
 print(neg[0])
 print(len(neg))
 print(neg[11784])
 print(len(noa))
 print(len(nov))
 print(noa[4065])
 print(nov[7718])
# counts of each rating
 cnt1=0
 cnt2=0
 cnt3=0
 cnt4=0
 cnt5=0
 for x in rating:
	if x==1:
		cnt1=cnt1+1
	elif x==2:
		cnt2=cnt2+1
	elif x==3:
		cnt3=cnt3+1
	elif x==4:
		cnt4=cnt4+1
	elif x==5:
		cnt5=cnt5+1

# gamma ratios for rated values[1-5]		
 gam5_1=cnt5/cnt1
 gam5_2=cnt5/cnt2
 gam5_3=cnt5/cnt3
 gam5_4=cnt5/cnt4
 gam5_5=cnt5/cnt5
 print(gam5_1)
 print(gam5_2)
 print(gam5_3)
 print(gam5_4)
 print(gam5_5)

# importing positive and negative words from csv files
 df1=pd.read_csv("C://Users/PrishitaRay/Documents/DM_Project/positive_words.csv")
 df2=pd.read_csv("C://Users/PrishitaRay/Documents/DM_Project/negative_words.csv")
 p=df1.values
 n=df2.values
 print(p[0][0])
 array1=df1.values
 array2=df2.values
 p=array1[:,0]
 n=array2[:,0]
#Counting occurrences of word and phrase tokens for each rating value(1-5) in review data
 d1={}
 d2={}
 d3={}
 cnt=[]
 C1=[]
 C2=[]
 C3=[]
 for i in range(0,len(neg)):
	l=[0,0,0,0,0]
	C1.append(l)
	
 for i in range(0,len(p)):
	l=[0,0,0,0,0]
	C2.append(l)
	
 for i in range(0,len(n)):
	l=[0,0,0,0,0]
	C3.append(l)
	
 for i in range(0,len(array)):
	for j in range(0,len(neg)):
		if neg[j] in rev_data[i]:
			if rating[i]==1:
				C1[j][0]=C1[j][0]+1
			elif rating[i]==2:
				C1[j][1]=C1[j][1]+1
			elif rating[i]==3:
				C1[j][2]=C1[j][2]+1
			elif rating[i]==4:
				C1[j][3]=C1[j][3]+1
			elif rating[i]==5:
				C1[j][4]=C1[j][4]+1
	for j in range(0,len(p)):
		if p[j] in rev_data[i]:
			if rating[i]==1:
				C2[j][0]=C2[j][0]+1
			elif rating[i]==2:
				C2[j][1]=C2[j][1]+1
			elif rating[i]==3:
				C2[j][2]=C2[j][2]+1
			elif rating[i]==4:
				C2[j][3]=C2[j][3]+1
			elif rating[i]==5:
				C2[j][4]=C2[j][4]+1
	for j in range(0,len(n)):
		if n[j] in rev_data[i]:
			if rating[i]==1:
				C3[j][0]=C3[j][0]+1
			elif rating[i]==2:
				C3[j][1]=C3[j][1]+1
			elif rating[i]==3:
				C3[j][2]=C3[j][2]+1
			elif rating[i]==4:
				C3[j][3]=C3[j][3]+1
			elif rating[i]==5:
				C3[j][4]=C3[j][4]+1

 print(C1[78780])
 print(C2[500])
 print(C3[500])
 cnt=0
 print(x)
 x=C1[0]
 print(x)
# remove negative phrase tokens having less than 10 total occurrences
 for x in C1:
	sum=0
	for j in range(0,5):
		sum=sum+x[j]
	if(sum<10):
		C1.remove(x)
		neg.remove(neg[cnt])
	else:
		cnt=cnt+1
		
 print(len(C1))
 print(len(neg))
 for i in range(0,20):
	print(C1[i])

 cnt=0
# positive word tokens
 pwt=[]
#negative word tokens
 nwt=[]
 for x in p:
	pwt.append(x)
 for x in n:
	nwt.append(x)		
 cnt=0
# remove positive word tokens having less than 10 total occurrences
 for x in C2:
	sum=0
	for j in range(0,5):
		sum=sum+x[j]
	if(sum<10):
		C2.remove(x)
		pwt.remove(pwt[cnt])
	else:
		cnt=cnt+1
		
 print(len(C2))
 print(C2[657])
 cnt=0
# remove negative word tokens having less than 10 total occurrences
 for x in C3:
	sum=0
	for j in range(0,5):
		sum=sum+x[j]
	if(sum<10):
		C3.remove(x)
		nwt.remove(nwt[cnt])
	else:
		cnt=cnt+1

 print(len(nwt))
 print(nwt[657])
 print(len(C3))
 print(C3[657])

#Sentiment Score Computation
 d1.fromkeys(neg)
 d2.fromkeys(pwt)
 d3.fromkeys(nwt)
 print(C1[1][0])
# Sentiment score analysis of negative phrase tokens
 for i in range(0,len(C1)):
	sum1=0
	sum2=1
	for j in range(1,6):
		if(j==1):
			sum1=sum1+(j*gam5_1*C1[i][j-1])
			sum2=sum2+(gam5_1*C1[i][j-1])
		elif(j==2):
			sum1=sum1+(j*gam5_2*C1[i][j-1])
			sum2=sum2+(gam5_2*C1[i][j-1])
		elif(j==3):
			sum1=sum1+(j*gam5_3*C1[i][j-1])
			sum2=sum2+(gam5_3*C1[i][j-1])
		elif(j==4):
			sum1=sum1+(j*gam5_4*C1[i][j-1])
			sum2=sum2+(gam5_4*C1[i][j-1])
		elif(j==5):
			sum1=sum1+(j*gam5_5*C1[i][j-1])
			sum2=sum2+(gam5_5*C1[i][j-1])
	if(sum2!=1):
		sum2=sum2-1
	ss=sum1/sum2
	d1[neg[i]]=ss

 print(d1[neg[8]])

# Sentiment score analysis of positive word tokens
 for i in range(0,len(C2)):
	sum1=0
	sum2=1
	for j in range(1,6):
		if(j==1):
			sum1=sum1+(j*gam5_1*C2[i][j-1])
			sum2=sum2+(gam5_1*C2[i][j-1])
		elif(j==2):
			sum1=sum1+(j*gam5_2*C2[i][j-1])
			sum2=sum2+(gam5_2*C2[i][j-1])
		elif(j==3):
			sum1=sum1+(j*gam5_3*C2[i][j-1])
			sum2=sum2+(gam5_3*C2[i][j-1])
		elif(j==4):
			sum1=sum1+(j*gam5_4*C2[i][j-1])
			sum2=sum2+(gam5_4*C2[i][j-1])
		elif(j==5):
			sum1=sum1+(j*gam5_5*C2[i][j-1])
			sum2=sum2+(gam5_5*C2[i][j-1])
	if(sum2!=1):
		sum2=sum2-1
	ss=sum1/sum2
	d2[pwt[i]]=ss

	
 print(d2[pwt[8]])

# Sentiment score analysis of negative word tokens
 for i in range(0,len(C3)):
	sum1=0
	sum2=1
	for j in range(1,6):
		if(j==1):
			sum1=sum1+(j*gam5_1*C3[i][j-1])
			sum2=sum2+(gam5_1*C3[i][j-1])
		elif(j==2):
			sum1=sum1+(j*gam5_2*C3[i][j-1])
			sum2=sum2+(gam5_2*C3[i][j-1])
		elif(j==3):
			sum1=sum1+(j*gam5_3*C3[i][j-1])
			sum2=sum2+(gam5_3*C3[i][j-1])
		elif(j==4):
			sum1=sum1+(j*gam5_4*C3[i][j-1])
			sum2=sum2+(gam5_4*C3[i][j-1])
		elif(j==5):
			sum1=sum1+(j*gam5_5*C3[i][j-1])
			sum2=sum2+(gam5_5*C3[i][j-1])
	if(sum2!=1):
		sum2=sum2-1
	ss=sum1/sum2
	d3[nwt[i]]=ss	
 print(d3[nwt[8]])

 sum=0
 for i in range(0,len(C1)):
	sum=sum+d1[neg[i]]

#  Mean of Sentiment Scores of negative phrase tokens	
 mean1=sum/len(C1)
 print(mean1)

 sum=0
 for i in range(0,len(C2)):
	sum=sum+d2[pwt[i]]

#  Mean of Sentiment Scores of positive word tokens	
 mean2=sum/len(C2)
 print(mean2)

 sum=0
 for i in range(0,len(C3)):
	sum=sum+d3[nwt[i]]

#  Mean of Sentiment Scores of negative word tokens
 mean3=sum/len(C3)
 print(mean3)

# Sentiment Scores of negative phrase tokens
 for i in range(0,5):
	print(d1[neg[i]])
	
# Sentiment Scores of positive word tokens
 for i in range(0,5):
	print(d2[pwt[i]])


# Sentiment Scores of negative word tokens
 for i in range(0,5):
	print(d3[nwt[i]])

#Feature Vector Formation
 print(type(neg))
 neg=list(d1.keys())
 pwt=list(d2.keys())
 nwt=list(d3.keys())
 feat_vec=[]
 for i in range(0,len(rev_data)):
	l=[]
	s1=""
	s2=""
	s3=""
	sump=0
	sumn=0
	nump=0
	numn=0
	for j in range(0,len(neg)):
		if neg[j] in rev_data[i]:
			s1+="1"
			sumn+=d1[neg[j]]
			numn+=1
		else:
			s1+="0"
	for j in range(0,len(pwt)):
		if pwt[j] in rev_data[i]:
			s2+="1"
			sump+=d2[pwt[j]]
			nump+=1
		else:
			s2+="0"
	for j in range(0,len(nwt)):
		if nwt[j] in rev_data[i]:
			s3+="1"
			sumn+=d3[nwt[j]]
			numn+=1
		else:
			s3+="0"
	if sump==0 and nump==0:
		ss_avgp=0
	else:
		ss_avgp=sump/nump
	if sumn==0 and numn==0:
		ss_avgn=0
	else:
		ss_avgn=sumn/numn
	ss_avg=ss_avgp-ss_avgn
	h1=hash(s1)
	h2=hash(s2)
	h3=hash(s3)
	l.append(h1)
	l.append(h2)
	l.append(h3)
	l.append(ss_avg)
	l.append(rating[i])
	sentences=rev_data[i].split(".")
	n=0
	m=0
	for x in sentences:
		cntn=0
		cntp=0
		for j in range(0,len(neg)):
			if neg[j] in x:
				cntn+=1
		for j in range(0,len(pwt)):
			if pwt[j] in x:
				cntp+=1
		for j in range(0,len(nwt)):
			if nwt[j] in x:
				cntn+=1
		if cntn>cntp:
			n+=1
		elif cntp>cntn:
			m+=1
	if m==n:
		val=0
	else:
		val=(-1*n)+m
	l.append(val)
	feat_vec.append(l)
 print(feat_vec[0])
 for x in feat_vec:
	x.remove(x[0])
	x.remove(x[0])
	x.remove(x[0])
	
 print(feat_vec[0])
 print(feat_vec[49999])
 for x in feat_vec:
	x.remove(x[0])

 for i in range(0,978780):
	feat_vec[i].append(rating[i])	
 print(feat_vec[49999])

#Classification Models
# List of feature vectors for training
 feat_vec_train=[]
 for i in range(0,900000):
	feat_vec_train.append(feat_vec[i])
 print(feat_vec_train[0])
 print(feat_vec[43657])
 print(feat_vec[47687])

# labels for supervised training of models
 labels=[]
 for i in range(0,900000):
	if(feat_vec[i][2]==4 or feat_vec[i][2]==5):
		labels.append('positive')
	elif feat_vec[i][2]==3:
		labels.append('neutral')
	elif(feat_vec[i][2]==1 or feat_vec[i][2]==2):
		labels.append('negative')

#1.Naïve Bayes Classifier 
#Naive Bayes Model	
 model=GaussianNB()
#Training the Naïve Bayes Model
 model.fit(feat_vec_train,labels)
GaussianNB(priors=None, var_smoothing=1e-09)
 test=[]

#List of feature vectors for testing or predicting
 for i in range(900000,978780):
	l=[]
	l.append(feat_vec[i])
	test.append(l)
 for i in range(0,20):
	print(test[i])

 result=[]
# Results predicted by Naïve Bayes Classifier
 for i in range(0,78780):
	result.append(model.predict(test[i]))	
 for i in range(0,20):
	print(result[i])

#2.Random Forest Classifier
#Random Forest Model
 clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
#Training the Random Forest Model
 clf.fit(feat_vec_train,labels)
# Feature Importances in predicting results identified by Random Forest Classifier
 print(clf.feature_importances_)
 result1=[]
#List of results predicted by Random Forest Classifier
 for i in range(0,78780):
	result1.append(clf.predict(test[i]))

 for i in range(0,20):
	print(result1[i])
	
#3.Support Vector Machines classifier
# SVM Model
 SvmModel=svm.SVC(gamma='scale', decision_function_shape='ovo') 
# Training the model
 SvmModel.fit(feat_vec_train,labels)
# Identified support vectors
 SvmModel.support_vectors_
 SvmModel.support_
 SvmModel.n_support_
 result2=[]
#List of Results Predicted by SVM classifier
 for i in range(0,78780):
	result2.append(SvmModel.predict(test[i]))
 for i in range(0,20):
	print(result2[i])


#Sentiment Polarity Categorization
 p1=0
 n1=0
 nt=0
 nt1=0
 p2=0
 n2=0
 nt2=0
 p3=0
 n3=0
 nt3=0

#Sentiment Polarity Categorization of Gaussian Naïve Bayes Classifier
 for i in range(0,78780):
	for l in result[i]:
		if l=='positive':
			p1=p1+1
		elif l=='negative':
			n1=n1+1
		elif l=='neutral':
			nt1=nt1+1

 p1=p1/78780
 n1=n1/78780
 nt1=nt1/78780
#percentage of positive reviews using Gaussian Naïve Bayes Classifier
 print(p1,’%’)
#percentage of negative reviews using Gaussian Naïve Bayes Classifier
 print(n1,’%’)
#percentage of neutral reviews using Gaussian Naïve Bayes Classifier
 print(nt1,’%’)

# Sentiment Polarity Categorization of Random Forest Classifier
 for i in range(0,78780):
	for l in result1[i]:
		if l=='positive':
			p2=p2+1
		elif l=='negative':
			n2=n2+1
		elif l=='neutral':
			nt2=nt2+1

 p2=p2/78780*100
 n2=n2/78780*100
 nt2=nt2/78780*100
#percentage of positive reviews using Random Forest Classifier
 print(p2,’%’)
#percentage of negative reviews using Random Forest Classifier
 print(n2,’%’)
#percentage of neutral reviews using Random Forest Classifier
 print(nt2,’%’)

#Sentiment Polarity Categorization of Support Vector Machines
 for i in range(0,78780):
	for l in result2[i]:
		if l=='positive':
			p3=p3+1
		elif l=='negative':
			n3=n3+1
		elif l=='neutral':
			nt3=nt3+1
		
 p3=p3/78780*100
 n3=n3/78780*100
 nt3=nt3/78780*100
#percentage of positive reviews using SVM Classifier
 print(p3,’%’)
#percentage of negative reviews using SVM Classifier
 print(n3,’%’)
#percentage of neutral reviews using SVM Classifier
 print(nt3,’%’)

# List of Correct results
 results_true=[]
 for i in range(900000,978780):
	l=[]
	if rating[i]==5 or rating[i]==4:
		l.append('positive')
	elif rating[i]==3:
		l.append('neutral')
	elif rating[i]==2 or rating[i]==1:
		l.append('negative')
	results_true.append(l)
	
 for i in range(0,20):
	print(results_true[i])

#F1 Scores
# f1_score of Gaussian Naïve Bayes Classifier
 f1_score(results_true, result, average='macro')
 f1_score(results_true,result,average='micro')

# f1_score of Random Forest classifier
 f1_score(results_true, result1, average='macro')
 f1_score(results_true, result1, average='micro')

# f1_score of Support Vector Machines classifier
 f1_score(results_true, result2, average='macro')
 f1_score(results_true, result2, average='micro')

y=[]
 for l in labels:
	if l=='positive':
		y.append(1)
	elif l=='negative':
		y.append(-1)
	elif l=='neutral':
		y.append(0)
		
 y=np.asarray(y)
 value=1.5
 width=0.75
 l=np.asarray(feat_vec_train)
 l=l.astype(np.float)
# Plotting the decision regions based on Support Vector Machine Model
 plot_decision_regions(X=l, y=y.astype(np.integer), clf=SvmModel,
                      filler_feature_values={2: value},
                      filler_feature_ranges={2: width}, legend=2)

