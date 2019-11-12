import requests
import csv
import xml.etree.ElementTree as ET
import os
pasta = "DTA/topics/"
caminhos = [os.path.join(pasta, nome) for nome in os.listdir(pasta)]
arquivos = [arq for arq in caminhos if os.path.isfile(arq)]
for topic in arquivos:
    aux = 0
    count = 0
    index = 0
    tp = topic.split('/')
    topicN = tp[2]
    # topicN = 'CD007394'
    list_label = []
    linha = ['Document Title','Abstract','Year','PDF Link','label']
    arq_content = open("DTA/qrels/full.train.dta.content.2019.qrels")
    arq_topic = open(topic,"r")
    # with open(nomeCSV, 'w') as file:     fazer isso pra gerar todos os titulos de topicos em um arquvio de texto
    for line in arq_topic:
        if 'Pids:' in line :
            list_of_pids = [[]]
            aux = 1
        elif aux == 1:
            line = line.lstrip()
            line = line.rstrip('\n')
            line = line.rstrip()
            list_of_pids[index].append(line)
            count = count + 1
        if count>300:
            index = index + 1
            list_of_pids.append([])
            count = 0
    arq_topic.close()
    for line in arq_content:
        if topicN in line:
            line = line.split(' ')
            if line[3][0:1] == '1':
                list_label.append(line[2])
    arq_content.close()
    if len(list_label) == 0:
        continue
    aux = 0
    j=0
    # print (list_of_pids)
    nomeCSV = '../workspace/data/' + topicN + '.csv'
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
#                           File "download_topic.py", line 61, in <module>
#     if ';' in child.text:
# TypeError: argument of type 'NoneType' is not iterable
# erro
                        # row[1] = row[1].encode('utf-8')
                        # row[0] = row[0].decode('utf-8')
                        # row[2] = row[2].encode('utf-8')
                        row[3] = child.text
                        if row[3] not in list_label:
                            row[4] = 'no'
                        else:
                            row[4] = 'yes'
                        j = j+1
                        if row[1] == '' :
                            row = ['','','','','0']
                            wtr = 1
                        else: 
                            writer.writerow(row)
                            row = ['','','','','0']
                            wtr = 1
    file.close()