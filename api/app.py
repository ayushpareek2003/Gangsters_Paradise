from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import predict

app=Flask(__name__)
cors=CORS(app,resources={r"*":{"origins":"*"}})
api=Api(app)

class Test(Resource):
    def get(self):
        return 'Ml walle aagye oey'
    def post(self):

        try:
            value=request.get_json()
            print(value)
            if(value):
                return {'Post Values': value},201
            
            return {"error":"Invalid format."}
        
        except Exception as error:
            return {"Error": error}
        

class Model(Resource):
    def get(self):
        return "<h1>hi</h1>"
    def post(self):
        try:
            value=request.get_json()
            
            if(value):
                text=value['des']

                return {'Post Values': predict.infer(text)},201
            
            return {"error":"Invalid format."}
        
        except Exception as error:
            return {"Error": error}


api.add_resource(Model,'/')

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0',port=port)
