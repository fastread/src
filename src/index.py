from __future__ import print_function, division

import os
import sys

root = os.getcwd().split("MAR")[0] + "MAR/src/util"
sys.path.append(root)

from flask import Flask, url_for, render_template, request, jsonify, Response, json
from pdb import set_trace
from mar import MAR

app = Flask(__name__,static_url_path='/static')


global target
target=MAR()

@app.route('/hello/')
def hello():
    return render_template('hello.html')


@app.route('/load',methods=['POST'])
def load():
    global target
    file=request.form['file']
    target=target.create(file)
    pos, neg, total = target.get_numbers()
    return jsonify({"hasLabel": target.hasLabel, "flag": target.flag, "pos": pos, "done": pos+neg, "total": total})

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

@app.route('/labeling',methods=['POST'])
def labeling():
    id = int(request.form['id'])
    label = request.form['label']
    target.code(id,label)
    pos, neg, total = target.get_numbers()
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
    pos,neg,total=target.get_numbers()
    random_id = target.random()
    res={"random": target.format(random_id)}
    if pos>0 and neg>0:
        uncertain_id, uncertain_prob, certain_id, certain_prob = target.train()
        res["certain"] = target.format(certain_id,certain_prob)
        res["uncertain"] = target.format(uncertain_id, uncertain_prob)
    target.save()
    # return jsonify(res)
    ress=json.dumps(res,ensure_ascii=False)
    response = Response(ress,content_type="application/json; charset=utf-8" )
    return response



if __name__ == "__main__":
    app.run(debug=False,use_debugger=False)