import requests
import csv
import xml.etree.ElementTree as ET
aux = 0
count = 0
index = 0
topic = "DTA/topics/CD007394"  #topico que vai ser escolhido
linha = ['Document Title','Abstract','Year','PDF Link','label']
arq_topic = open(topic,"r")
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
aux = 0
# print (list_of_pids)
with open('output.csv', 'w') as file:
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
                    row[1] = row[1].encode('utf-8')
                    row[0] = row[0].encode('utf-8')
                    row[2] = row[2].encode('utf-8')
                    row[3] = child.text
                    writer.writerow(row)
                    row = ['','','','','0']
                    wtr = 1
file.close()