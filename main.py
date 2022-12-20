import cv2
import numpy as np
from functools import reduce
import requests


class DigitRecognizer:
    def __init__(self, model) -> None:
        self.model = model


    def execute_features(self, path: str)-> tuple:
        """Execute features from the link and uses this data for model training.
        - Accepts link to the picture with training data
        - Returns tuple of features and target for model"""

        im = cv2.imread(path)
        try:
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            thresh = cv2.adaptiveThreshold(blur,255,1,1,25,2,)
        except cv2.error as e:
            print(e)
            for k in dir(e):
                if k[0:2] != "__":
                    print("e.%s = %s" % (k, getattr(e, k)))

                # handle error: empty img
                if e.err == "!_src.empty()":
                    return  #       

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

        samples =  np.empty((0,100))
        samples_x = []

        responses = np.array([1, 2, 10, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 10, 3, 4])

        for cnt in contours:
            if cv2.contourArea(cnt)>50:
                [x,y,w,h] = cv2.boundingRect(cnt)
                
                if  h>12:
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
        
                    sample = roismall.reshape((1,100))
                    samples = np.append(samples,sample,0)
                    samples_x.append(x)

        samples_resp_dict = dict(zip(samples_x, samples))
        samples = np.array(list(dict(sorted(samples_resp_dict.items())).values()))

        return samples, responses


    def train(self, train_path: str, layout=cv2.ml.ROW_SAMPLE)-> None:
        # Load the training img
        p = requests.get(train_path)
        with open("img.jpg", "wb") as f:
            f.write(p.content)
            f.close()

        samples, responses = self.execute_features("img.jpg")
        data = samples.astype(np.float32)

        self.model.train(data, responses=responses, layout=layout)


    def test(self, im) -> str:
        """- Accepts image and trained model as arguments
           - Returns numbers read from the image"""

        try:
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
        except cv2.error as e:
            print(e)
            for k in dir(e):
                if k[0:2] != "__":
                    print("e.%s = %s" % (k, getattr(e, k)))

                # handle error: empty img
                if e.err == "!_src.empty()":
                    return  #

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)

        res = []
        res_x = []

        for cnt in contours:
            if cv2.contourArea(cnt)>50:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>28:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                    roi = thresh[y:y+h,x:x+w]
                    roismall = cv2.resize(roi,(10,10))
                    roismall = roismall.reshape((1,100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
                    if int((results[0][0])) == 10:
                        res.append(",")
                    else:
                        res.append(int((results[0][0])))
                    res_x.append(x)      

        res_dict = dict(zip(res_x, res))
        numbs = list(dict(sorted(res_dict.items())).values())

        res = reduce(lambda x,y: str(x) + str(y), numbs)

        return res
   

if __name__ == "__main__":
    # Load the data for training
    train_path = "https://stepik.org/media/attachments/course/132193/training_data.jpg"

    # Define the model
    model = cv2.ml.KNearest_create()
    dr = DigitRecognizer(model)
    
    # Training part
    dr.train(train_path)

    # Testing part
    test_path = "imgs/1.png"

    im = cv2.imread(test_path)
    print(dr.test(im))
