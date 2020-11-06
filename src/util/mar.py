from __future__ import print_function, division
try:
    import cPickle as pickle
except:
    import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
from sklearn import linear_model
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

class MAR(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 10
        self.enough = 30
        self.kept=50
        self.atleast=100


    def create(self,filename):
        self.filename=filename
        self.name=self.filename.split(".")[0]
        self.flag=True
        self.hasLabel=True
        self.record={"x":[],"pos":[]}
        self.body={}
        self.est = []
        self.est_num = 0
        self.last_pos=0
        self.last_neg=0


        try:
            ## if model already exists, load it ##
            self = self.load()
        except:
            ## otherwise read from file ##
            try:
                self.loadfile()
                self.preprocess()
                self.save()
            except:
                ## cannot find file in workspace ##
                self.flag=False
        self.enable_est=False
        return self

    ### Depreciated
    ### Use previous knowledge, labeled only
    def create_old(self, filename):
        with open("../workspace/coded/" + str(filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "code", "time"]
        header = content[0]
        ind0 = header.index("code")
        self.last_pos = len([c[ind0] for c in content[1:] if c[ind0] == "yes"])
        self.last_neg = len([c[ind0] for c in content[1:] if c[ind0] == "no"])
        for field in fields:
            ind = header.index(field)
            if field == "time":
                self.body[field].extend([float(c[ind]) for c in content[1:] if c[ind0] != "undetermined"])
            else:
                self.body[field].extend([c[ind] for c in content[1:] if c[ind0] != "undetermined"])
        try:
            ind = header.index("label")
            self.body["label"].extend([c[ind] for c in content[1:] if c[ind0]!="undetermined"])
        except:
            self.body["label"].extend(["unknown"] * (len([c[ind0] for c in content[1:] if c[ind0]!="undetermined"])))
        try:
            ind = header.index("fixed")
            self.body["fixed"].extend([c[ind] for c in content[1:] if c[ind0]!="undetermined"])
        except:
            self.body["fixed"].extend([0] * (len([c[ind0] for c in content[1:] if c[ind0]!="undetermined"])))

        self.preprocess()
        self.save()


    def loadfile(self):
        self.body = pd.read_csv("../workspace/data/" + str(self.filename),encoding = "ISO-8859-1")
        fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        columns = self.body.columns
        n = len(self.body)
        for field in fields:
            if field not in columns:
                self.body[field] = [""]*n
        if "label" not in columns:
            self.body["label"] = ["unknown"]*n
        if "code" not in columns:
            self.body["code"] = ["undetermined"]*n
        if "time" not in columns:
            self.body["time"] = [0]*n
        if "fixed" not in columns:
            self.body["fixed"] = [0]*n
        self.body = self.body.fillna("")
        return

    def export_feature(self):
        with open("../workspace/coded/feature_" + str(self.name) + ".csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for i in range(self.csr_mat.shape[0]):
                for j in range(self.csr_mat.indptr[i],self.csr_mat.indptr[i+1]):
                    csvwriter.writerow([i+1,self.csr_mat.indices[j]+1,self.csr_mat.data[j]])
        return

    def get_numbers(self):
        total = len(self.body["code"]) - self.last_pos - self.last_neg
        pos = Counter(self.body["code"])["yes"] - self.last_pos
        neg = Counter(self.body["code"])["no"] - self.last_neg
        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total

    def export(self):
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "label", "code","time"]
        body = self.body[fields]
        body.sort_values(by=['time'], ascending=False)
        yes = body.loc[body['code'] == 'yes']
        no = body.loc[body['code'] == 'no']
        und = body.loc[body['code'] == 'undetermined']
        out = pd.concat([yes, no, und], ignore_index=True)
        out.to_csv("../workspace/coded/" + str(self.name) + ".csv",columns=fields,index=False)
        return

    def preprocess(self):
        ### Combine title and abstract for training ##################
        content = [str(self.body["Document Title"][index]) + " "  + str(self.body["Abstract"][index]) for index in range(len(self.body))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore")
        tfidf = tfidfer.fit_transform(content)

        weight = tfidf.sum(axis=0).tolist()[0]
        kept = np.argsort(weight)[-self.fea_num:]
        self.voc = np.array(list(tfidfer.vocabulary_.keys()))[np.argsort(list(tfidfer.vocabulary_.values()))][kept]

        ##############################################################

        ### Term frequency as feature, L2 normalization ##########
        tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=u'l2', use_idf=False,
                        vocabulary=self.voc,decode_error="ignore")
        # tfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False,
        #                 vocabulary=self.voc,decode_error="ignore")
        self.csr_mat=tfer.fit_transform(content)
        ########################################################
        return

    ## save model ##
    def save(self):
        with open("memory/"+str(self.name)+".pickle","wb") as handle:
            pickle.dump(self,handle)


    ## load model ##
    def load(self):
        with open("memory/" + str(self.name) + ".pickle", "rb") as handle:
            tmp = pickle.load(handle)
        return tmp


    def estimate_curve(self, clf, reuse=False, num_neg=0):

        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= 1:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    count -= 1
                    can = []
            return sample







        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        ###############################################

        # prob = clf.predict_proba(self.csr_mat)[:,:1]
        prob1 = clf.decision_function(self.csr_mat)
        prob = np.array([[x] for x in prob1])
        # prob = self.csr_mat


        y = np.array([1 if x == 'yes' else 0 for x in self.body['code']])
        y0 = np.copy(y)

        if len(poses) and reuse:
            all = list(set(poses) | set(negs) | set(self.pool))
        else:
            all = range(len(y))


        pos_num_last = Counter(y0)[1]

        lifes = 3
        life = lifes


        while (True):
            C = Counter(y[all])[1]/ num_neg
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True, C=C)

            es.fit(prob[all], y[all])
            pos_at = list(es.classes_).index(1)


            pre = es.predict_proba(prob[self.pool])[:, pos_at]


            y = np.copy(y0)

            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1



            pos_num = Counter(y)[1]

            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num


        esty = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]

        return esty, pre




    ## Train model ##
    def train(self,pne=True,weighting=True):

        clf = svm.SVC(kernel='linear', probability=True, class_weight='balanced') if weighting else svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),2*len(left),self.atleast)),replace=False)
        except:
            pass

        if not pne:
            unlabeled=[]

        labels=np.array([x if x!='undetermined' else 'no' for x in self.body['code']])
        all_neg=list(negs)+list(unlabeled)
        sample = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[sample], labels[sample])


        ## aggressive undersampling ##
        if len(poses)>=self.enough:

            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
        elif pne:
            train_dist = clf.decision_function(self.csr_mat[unlabeled])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            unlabel_sel = np.argsort(train_dist)[::-1][:int(len(unlabeled) / 2)]
            sample = list(decayed) + list(np.array(unlabeled)[unlabel_sel])
            clf.fit(self.csr_mat[sample], labels[sample])


        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        if self.enable_est:
            if self.last_pos>0 and len(poses)-self.last_pos>0:
                self.est_num, self.est = self.estimate_curve(clf, reuse=True, num_neg=len(sample)-len(left))
            else:
                self.est_num, self.est = self.estimate_curve(clf, reuse=False, num_neg=len(sample)-len(left))
            return uncertain_id, self.est[uncertain_id], certain_id, self.est[certain_id], clf
        else:
            return uncertain_id, uncertain_prob, certain_id, certain_prob, clf

    ## reuse
    def train_reuse(self,pne=True):
        pne=True
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        if len(left)==0:
            return [], [], self.random(), []



        decayed = list(left) + list(negs)

        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(decayed), self.atleast)), replace=False)
        except:
            pass

        if not pne:
            unlabeled = []


        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        sample = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[sample], labels[sample])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
        elif pne:
            train_dist = clf.decision_function(self.csr_mat[unlabeled])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist = -train_dist
            unlabel_sel = np.argsort(train_dist)[::-1][:int(len(unlabeled) / 2)]
            sample = list(decayed) + list(np.array(unlabeled)[unlabel_sel])
            clf.fit(self.csr_mat[sample], labels[sample])

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)

        if self.enable_est:
            self.est_num, self.est = self.estimate_curve(clf, reuse=False, num_neg=len(sample)-len(left))
            return uncertain_id, self.est[uncertain_id], certain_id, self.est[certain_id], clf
        else:
            return uncertain_id, uncertain_prob, certain_id, certain_prob, clf



    ## Get suspecious codes
    def susp(self,clf):
        thres_pos = 1
        thres_neg = 0.5
        length_pos = 10
        length_neg = 10

        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        # poses = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        # negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        poses = np.array(poses)[np.where(np.array(self.body['fixed'])[poses] == 0)[0]]
        negs = np.array(negs)[np.where(np.array(self.body['fixed'])[negs] == 0)[0]]

        if len(poses)>0:
            pos_at = list(clf.classes_).index("yes")
            prob_pos = clf.predict_proba(self.csr_mat[poses])[:,pos_at]
            # se_pos = np.argsort(prob_pos)[:length_pos]
            se_pos = np.argsort(prob_pos)
            # se_pos = [s for s in se_pos if prob_pos[s]<thres_pos]
            sel_pos = poses[se_pos]
            probs_pos = prob_pos[se_pos]
        else:
            sel_pos = np.array([])
            probs_pos = np.array([])

        if len(negs)>0:
            if clf:
                neg_at = list(clf.classes_).index("no")
                prob_neg = clf.predict_proba(self.csr_mat[negs])[:,neg_at]
                # se_neg = np.argsort(prob_neg)[:length_neg]
                se_neg = np.argsort(prob_neg)
                # se_neg = [s for s in se_neg if prob_neg[s]<thres_neg]
                sel_neg = negs[se_neg]
                probs_neg = prob_neg[se_neg]
            else:
                sel_neg = negs
                probs_neg = np.array([])
        else:
            sel_neg = np.array([])
            probs_neg = np.array([])

        return sel_pos, probs_pos, sel_neg, probs_neg

    ## BM25 ##
    def BM25(self,query):
        b=0.75
        k1=1.5

        ### Combine title and abstract for training ###########
        content = [str(self.body["Document Title"][index]) + " " + str(self.body["Abstract"][index]) for index in
                   range(len(self.body["Document Title"]))]
        #######################################################

        ### Feature selection by tfidf in order to keep vocabulary ###

        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=False, smooth_idf=False,
                                  sublinear_tf=False, decode_error="ignore")
        tf = tfidfer.fit_transform(content)
        d_avg = np.mean(np.sum(tf, axis=1))
        score = {}
        for word in query:
            score[word]=[]
            id= tfidfer.vocabulary_[word]
            df = sum([1 for wc in tf[:,id] if wc>0])
            idf = np.log((len(content)-df+0.5)/(df+0.5))
            for i in range(len(content)):
                score[word].append(idf*tf[i,id]/(tf[i,id]+k1*((1-b)+b*np.sum(tf[0],axis=1)[0,0]/d_avg)))
        self.bm = np.sum(list(score.values()),axis=0)

    def BM25_get(self):
        ids = self.pool[np.argsort(self.bm[self.pool])[::-1][:self.step]]
        scores = self.bm[ids]
        return ids, scores

    ## Get certain ##
    def certain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        if len(self.pool)==0:
            return [],[]
        prob = clf.predict_proba(self.csr_mat[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1][:self.step]
        return np.array(self.pool)[order],np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self,clf):
        pos_at = list(clf.classes_).index("yes")
        if len(self.pool)==0:
            return [],[]
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        train_dist = clf.decision_function(self.csr_mat[self.pool])
        order = np.argsort(np.abs(train_dist))[:self.step]  ## uncertainty sampling by distance to decision plane
        # order = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)

    ## Format ##
    def format(self,id,prob=[]):
        result=[]
        for ind,i in enumerate(id):
            tmp = {key: str(self.body[key][i]) for key in self.body}
            tmp["id"]=str(i)
            if prob!=[]:
                tmp["prob"]=prob[ind]
            result.append(tmp)
        return result

    ## Code candidate studies ##
    def code(self,id,label):
        if self.body['code'][id] == label:
            self.body['fixed'][id] = 1
        self.body["code"][id] = label
        self.body["time"][id] = time.time()


    ## Plot ##
    def plot(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        if len(self.labeled)<=0:
            return

        fig = plt.figure()
        order = np.argsort(np.array(self.body['time'])[self.labeled])
        seq = np.array(self.body['code'])[np.array(self.labeled)[order]]
        counter = 0
        rec = [0]
        for s in seq:
            if s=='yes':
                counter+=1
            rec.append(counter)
        plt.plot(range(len(rec)), rec)


        plt.ylabel("Relevant Found")
        plt.xlabel("Documents Reviewed")
        name=self.name+ "_" + str(int(time.time()))+".png"

        dir = "./static/image"
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

        plt.savefig("./static/image/" + name)
        plt.close(fig)
        return name

    def get_allpos(self):
        return len([1 for c in self.body["label"] if c=="yes"])-self.last_pos

    ## Restart ##
    def restart(self):
        os.remove("./memory/"+self.name+".pickle")

    ## Get missed relevant docs ##
    def get_rest(self):
        rest=[x for x in range(len(self.body['label'])) if self.body['label'][x]=='yes' and self.body['code'][x]!='yes']
        rests={}
        # fields = ["Document Title", "Abstract", "Year", "PDF Link"]
        fields = ["Document Title"]
        for r in rest:
            rests[r]={}
            for f in fields:
                rests[r][f]=self.body[f][r]
        return rests

    def latest_labeled(self):
        order = np.argsort(np.array(self.body['time'])[self.labeled])[::-1]
        return np.array(self.labeled)[order]