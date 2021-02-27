from flask import Flask, request, jsonify
import Algo6

app = Flask(__name__)

@app.route('/store', methods=["GET", "POST"])
def dregister():
    if request.method == 'GET':
        return jsonify("This is get method")
    else:
        r = request.json['review']
        p = request.json['item_type']
        Algo6.api(r,p)
        return jsonify(r,p)

if __name__ == '__main__':
    app.run(debug=True)