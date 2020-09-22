from __future__ import print_function, division

import os
import sys
sys.path.append('./util/')

from flask import Flask, url_for, render_template, request, jsonify, Response, json
from pdb import set_trace
from mar import MAR

app = Flask(__name__,static_url_path='/static')

global target
target=MAR()
global clf
clf = []

@app.route('/')
def hello():
    return render_template('hello.html')


@app.route('/load',methods=['POST'])
def load():
    global target
    file=request.form['file']
    target=target.create(file)
    pos, neg, total = target.get_numbers()
    return jsonify({"hasLabel": target.hasLabel, "flag": target.flag, "pos": pos, "done": pos+neg, "total": total})

# Depreciated
@app.route('/load_old',methods=['POST'])
def load_old():
    global target
    file=request.form['file']
    target.create_old(file)
    if target.last_pos==0:
        target.flag=False
    return jsonify({"flag": target.flag})

@app.route('/export',methods=['POST'])
def export():
    try:
        target.export()
        flag=True
    except:
        flag=False
    return jsonify({"flag": flag})

@app.route('/plot',methods=['POST'])
def plot():
    dir = "./static/image"
    for file in os.listdir(dir):
        os.remove(os.path.join(dir,file))
    name = target.plot()
    return jsonify({"path": name})

@app.route('/est',methods=['POST'])
def est():
    stat = request.form['stat']
    if stat == 'true':
        target.enable_est = True
    else:
        target.enable_est = False
    return jsonify({})

@app.route('/labeling',methods=['POST'])
def labeling():
    id = int(request.form['id'])
    label = request.form['label']
    target.code(id,label)
    pos, neg, total = target.get_numbers()
    target.save()
    return jsonify({"flag": target.flag, "pos": pos, "done": pos + neg, "total": total})

@app.route('/auto',methods=['POST'])
def auto():
    for id in request.form.values():
        target.code(int(id),target.body["label"][int(id)])
    pos, neg, total = target.get_numbers()
    return jsonify({"flag": target.flag, "pos": pos, "done": pos + neg, "total": total})

@app.route('/restart',methods=['POST'])
def restart():
    global target
    os.remove("./memory/"+target.name+".pickle")
    target = target.create(target.filename)
    pos, neg, total = target.get_numbers()
    return jsonify({"hasLabel": target.hasLabel, "flag": target.flag, "pos": pos, "done": pos + neg, "total": total})

@app.route('/train',methods=['POST'])
def train():
    global clf
    pos,neg,total=target.get_numbers()
    res={}
    if pos>0 or target.last_pos>0:
        uncertain_id, uncertain_prob, certain_id, certain_prob, clf = target.train(pne=True)
        res["uncertain"] = target.format(uncertain_id,uncertain_prob)
        res["certain"] = target.format(certain_id,certain_prob)
        # if target.last_pos > 0 and pos > 0:
        #     uncertain_id, uncertain_prob, certain_reuse_id, certain_reuse_prob = target.train_reuse(pne=True)
        #     res["reuse"] = target.format(certain_reuse_id, certain_reuse_prob)
    else:
        random_id = target.random()
        res["uncertain"] = target.format(random_id)
    if target.enable_est:
        res['est'] = target.est_num
    target.save()
    return jsonify(res)

@app.route('/susp',methods=['POST'])
def susp():
    res={}
    pos_id, pos_prob, neg_id, neg_prob = target.susp(clf)
    latest_id = target.latest_labeled()
    res["pos"] = target.format(pos_id,pos_prob)
    res["neg"] = target.format(neg_id,neg_prob)
    res["latest"] = target.format(latest_id)
    return jsonify(res)

@app.route('/search',methods=['POST'])
def search():
    import re
    query=request.form['query']
    keywords = re.sub(r'[\W_]+', ' ', query).split()
    target.BM25(keywords)
    res={}
    ids, scores = target.BM25_get()
    res['bm25']=target.format(ids,scores)
    return jsonify(res)



if __name__ == "__main__":
    app.run(debug=False,use_debugger=False)
