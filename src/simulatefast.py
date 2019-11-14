from mar import MAR
import numpy as np
import math, os, sys
import os


numberrel=[62,104,48,45]

def BM25(filename, seed, stop, query='',  stopat=0.95, starting=1, thres=30,j=0):

   stopat = float(stopat)

   np.random.seed(seed)

   read = MAR()
   
   read = read.create(filename)
   #read.restart()
   # query for BM25 to boost the initial retrieval. Example query='defect prediction'
   print(query)
   read.BM25(query.strip().split(' '))

   num2 = read.get_allpos()

   target = int(num2 * stopat)

   # whether to estimate recall

   if stop == 'est':

       read.enable_est = True

   else:

       read.enable_est = False

   while True:

       pos, neg, total = read.get_numbers()

       # try:

       print("%d, %d, %d" %(pos, pos+neg, read.est_num))

       # except:

       # print("%d, %d" %(pos, pos+neg))

       #if pos + neg >= numberrel[j]:

           #break

       if pos < starting:

           for id in read.BM25_get():
               print(id)
               read.code(id, read.body['label'][id])

       else:

           a,b,c,d,_ =read.train(weighting=True, pne=True)

           # stop if estimated recall is above the target recall

           if stop == 'est':
               print ("est execution")  
               if stopat * read.est_num <= pos:

                   break

           # Cormack16 stopping rule

           elif stop == 'knee':
               print ("knee execution") 
               if pos>=thres:

                   if read.knee():

                       break

           # stop if true recall is above the target recall

           else:

               if pos >= target:

                   break

           # uncertainty sampling

           if pos < thres:

               for id in a:

                   read.code(id, read.body['label'][id])

           # certainty sampling

           else:

               for id in c:

                   read.code(id, read.body['label'][id])

   # read.export()

   return read


 
def calcular(valores=None, calculos=None):
    if valores:
        if valores.__class__.__name__ == 'list' and calculos.__class__.__name__ == 'dict':
            def somar(valores):
                soma = 0
                for v in valores:
                    soma += v
                return soma
 
 
            def media(valores):
                soma = somar(valores)
                qtd_elementos = len(valores)
                media = soma / float(qtd_elementos)
                return media
 
 
            def variancia(valores):
                _media = media(valores)
                soma = 0
                _variancia = 0
 
                for valor in valores:
                    soma += math.pow( (valor - _media), 2)
                _variancia = soma / float( len(valores) )
                return _variancia
 
 
            def desvio_padrao(valores):
                return math.sqrt( variancia(valores) )
                 
 
            calculos['soma'] = somar(valores)
            calculos['media'] = media(valores)
            calculos['variancia'] = variancia(valores)
            calculos['desvio_padrao'] = desvio_padrao(valores)

import nltk
# nltk.download() se precisar baixar o nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
simbols = ['(',')','-']
suffix="_fast)_fast_KNEE"
pasta = "DTA/topics/"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
query=[]
for topic in arquivos:
    arq_topic = open(topic,"r")
    for line in arq_topic:
        if 'Title:' in line:
            line = line.lstrip('Title: ')
            line = line.rstrip(' \n')
            line = unicode(line, 'utf-8')
            line = line.lower()
            line = line.encode('utf-8')
            word_tokens = word_tokenize(line) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words and not w in simbols]
            line = ' '.join(filtered_sentence)
            query.append(line)
    arq_topic.close()
arquivos = [arq.split('/')[2]+'.csv' for arq in arquivos]
filesInput=arquivos
filesOutput=[a.rstrip('.csv') for a in arquivos]
# filesInput = ['CD012669.csv']
# filesOutput = ['CD012669']
# query = ['point care ultrasonography diagnosing thoracoabdominal injuries patients blunt trauma']
for j in range(len(arquivos)):
    filew=open(filesOutput[j],"w") 
    queryw=query[j]
    media_recal=0
    media_precision=0
    media_posit=0
    media_all=0
    rangevalue=1
    recalList=[]
    precisionList=[]
    positList=[]
    allList=[]
    for i in range (rangevalue):
        read=BM25(filesInput[j], i, 'knee', queryw,0.95,1,30,j)
        filew.write(("posit "+ str(read.record['pos'][-1])+ " all "+ str(read.record['x'][-1])) +"\n")
        filew.write(str(read.get_numbers())+"\n")
        filew.write(("recall:"+ str(read.record['pos'][-1]/float(read.get_allpos()))+ " precision " + str(read.record['pos'][-1]/float(read.record['x'][-1])))+"\n")
        media_recal+=(read.record['pos'][-1]/float(read.get_allpos()))
        media_precision+=read.record['pos'][-1]/float(read.record['x'][-1])
        media_posit+=(read.record['pos'][-1])
        media_all+=read.record['x'][-1]
        recalList.append((read.record['pos'][-1]/float(read.get_allpos())))
        precisionList.append(read.record['pos'][-1]/float(read.record['x'][-1]))
        positList.append((read.record['pos'][-1]))
        allList.append(read.record['x'][-1])
        print (recalList)
        filew.flush()
        
    
    calculos = {}
    calcular(recalList, calculos)    
    filew.write("recall calculos['soma'] "+ str(calculos['soma']) +" calculos['media'] " + str(calculos['media']) +" calculos['variancia'] "+ str(calculos['variancia']) +" calculos['desvio_padrao'] " + str(calculos['desvio_padrao'])+"\n")
    calcular(precisionList, calculos)    
    filew.write("precision calculos['soma'] "+ str(calculos['soma']) +" calculos['media'] " + str(calculos['media']) +" calculos['variancia'] "+ str(calculos['variancia']) +" calculos['desvio_padrao'] " + str(calculos['desvio_padrao'])+"\n")            
    calcular(allList, calculos)    
    filew.write("allList calculos['soma'] "+ str(calculos['soma']) +" calculos['media'] " + str(calculos['media']) +" calculos['variancia'] "+ str(calculos['variancia']) +" calculos['desvio_padrao'] " + str(calculos['desvio_padrao'])+"\n")            
    filew.write(str(media_recal/float(rangevalue))+"\n")
    filew.write(str(media_precision/float(rangevalue))+"\n")
    filew.write(str(media_posit/float(rangevalue))+"\n")
    filew.write(str(media_all/float(rangevalue))+"\n")
    
    filew.close()
