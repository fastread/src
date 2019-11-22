import requests
import csv
import xml.etree.ElementTree as ET
import os
pasta = "DTA/topics/"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
count = 0
index = 0
aux = 0
list_of_pids = [[]]
list_topics = []
for topic in arquivos:
    arq_topic = open(topic,"r")
    for line in arq_topic:
        if 'Pids:' in line :
            aux = 1
        elif aux == 1:
            line = line.lstrip()
            line = line.rstrip('\n')
            line = line.rstrip()
            if line not in list_topics:
                list_topics.append(line)
    arq_topic.close()
    aux = 0
    print (len(list_topics))

for doc in list_topics:
    if count > 300:
        index = index + 1
        list_of_pids.append([])
        count = 0
    list_of_pids[index].append(doc)
    count = count + 1

linha = ['Document Title','Abstract','Year','PDF Link','label']
nomeCSV = '../workspace/data/teste.csv'
with open(nomeCSV, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(linha)
    for i in range(len(list_of_pids)):
        payload = {'db': 'pubmed', 'id': list_of_pids[i], 'rettype': 'xml', 'retmode': 'xml'}
        r = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?', params=payload)
        tree =  ET.ElementTree(ET.fromstring(r.content))
        root = tree.getroot()
        filtro = "*"
        row = ['','','','','0']
        wtr = 0
        aux2 = ''
        for child in root.iter(filtro):
            # print(child.tag,child.text)   
            # se colocar esse print de cima vai ver o conteudo do xml, ai pode ver que tem abstracts faltando
            if child.tag == 'ArticleTitle':
                wtr = 0
                if child.text == None:
                    continue
                if ';' in child.text:
                    aux2 = child.text.split(';')
                    child.text = aux2[0] + aux2[1]
                row[0] = child.text         
            elif child.tag == 'AbstractText':
                if child.text == None:
                    continue
                if ';' in child.text:
                    aux2 = child.text.split(';')
                    child.text = aux2[0] + aux2[1]
                row[1]= row[1] + child.text
                # print(row)
            elif child.tag == 'Year':
                row[2] = child.text
            elif child.tag == 'ArticleId':
                if wtr == 0:
                    row[3] = child.text
                    # if row[3] not in list_label:
                    #     row[4] = 'no'
                    # else:
                    #     row[4] = 'yes'
                    # j = j+1
                    if row[1] == '' :
                        row = ['','','','','0']
                        wtr = 1
                    else: 
                        writer.writerow(row)
                        row = ['','','','','0']
                        wtr = 1
file.close()