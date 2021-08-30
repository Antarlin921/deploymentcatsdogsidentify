from keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model(r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\CatsDogs\flask_app\Newfolder\catdog.h5')

# image path
# input  image location that is to be classified
img_path=r'C:\Users\Antarlin\Desktop\Data_Science\Deployment26July2021\CatsDogs\test_set\test_set\dogs\dog.4026.jpg'

# read the image
test_image=image.load_img(img_path,target_size=(64,64))


# image to array
test_image=image.img_to_array(test_image)
test_image=test_image.reshape(1,64,64,3)
result=model.predict(test_image)
if [result[0]>0.5][0]==True:
  print("dog")
else:
  print("cat")

# to check the classes

# training_set.class_indices