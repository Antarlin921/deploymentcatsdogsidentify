# import a library 

from flask import Flask,render_template,request
# from flask import Flask ,render_template,request
import contextvars
# Flask ,render_template,request
import joblib

from keras.models import load_model
from tensorflow.keras.preprocessing import image

# instace of an app
app=Flask(__name__)
model=load_model(r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\CatsDogs\flask_app\catdog.h5')


@app.route('/')
def hello():
    return "Welcome"

@app.route('/home')
def homepage():
    return render_template("home.html")



#Picture Detection Code Start
@app.route('/blog1',methods=['POST'])
def contact1():  
    piclink= request.form.get('Picture_Link')

    print(piclink)

    #  image path
    # img_path=r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\CatsDogs\test_set\test_set\dogs\dog.4028.jpg'
    # img_path='C:/Users/Antarlin/Desktop/Data_Science/Deployment26July2021/CatsDogs/test_set/test_set/dogs/dogs.4027.jpg'
    # img_path=r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\CatsDogs\Joblib\dog_4027.jpg'
    # read the image
    test_image=image.load_img(piclink,target_size=(64,64))


    # image to array
    test_image=image.img_to_array(test_image)
    test_image=test_image.reshape(1,64,64,3)
    result=model.predict(test_image)
    if [result[0]>0.5][0]==True:
        output="dog"
    else:
        output="cat"
    return render_template('result.html',predicted_text=f'This is a {result,output}')

#  run the app  
if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)

