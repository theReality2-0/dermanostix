from flask import Flask, render_template, Response
import cv2
from dmodel import Lesions


app = Flask(__name__)
model = Lesions()
ob = cv2.CascadeClassifier('cascade.xml')

@app.route('/')
def function():
    return render_template('buttons.html')


@app.route('/video_stream')
def video_stream():
    return render_template('detec.html')

font = cv2.FONT_HERSHEY_SIMPLEX
cf = ""
take = 0

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        global cf
        _, fr = self.video.read()
        gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        obs = ob.detectMultiScale(gr, 6, 1)
        for (x, y, w, h) in obs:
            fc = fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (224, 224))
            pred = model.make_prediction(roi)
            cf = pred
            cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0,0), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/diagnose')
def diagnose():
    site = {
        "Melanocytic Nevi": "https://www.mayoclinic.org/diseases-conditions/moles/symptoms-causes/syc-20375200",
        "Melanoma": "https://www.mayoclinic.org/diseases-conditions/melanoma/symptoms-causes/syc-20374884",
        "Seborrheic Keratosis": "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878",
        "Basal Cell Carcinoma": "https://www.mayoclinic.org/diseases-conditions/basal-cell-carcinoma/symptoms-causes/syc-20354187",
        "Actinic Keratosis": "https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969",
        "Vascular Lesion": "https://www.google.com/search?q=vascular+lesion&rlz=1C5CHFA_enUS746US746&oq=vascular+lesion&aqs=chrome..69i57j0l4j69i60l3.1959j0j7&sourceid=chrome&ie=UTF-8",
        "Dermatofibroma": "https://www.webmd.com/skin-problems-and-treatments/picture-of-dermatofibroma"
    }

    blurbs = {
        "Melanocytic Nevi": """A usually noncancerous disorder of pigment-producing skin cells commonly called birth 
                            marks or moles. This type of mole is often large and caused by a disorder involving 
                            melanocytes, cells that produce pigment (melanin). Melanocytic nevi can be rough, flat, 
                            or raised. They can exist at birth or appear later. Rarely, melanocytic nevi can become cancerous. 
                            Most cases don't require treatment, but some cases require removal of the mole. (Google)""",
        "Melanoma": """The most serious type of skin cancer.
                    Melanoma occurs when the pigment-producing cells that give color to the skin become cancerous.
                    Symptoms might include a new, unusual growth or a change in an existing mole. Melanomas can occur anywhere on the body.
                    Treatment may involve surgery, radiation, medications, or in some cases chemotherapy. (Google)""",
        "Seborrheic Keratosis": """A noncancerous skin condition that appears as a waxy brown, black, or tan growth.
                                A seborrheic keratosis is one of the most common noncancerous skin growths in older 
                                adults. While it's possible for one to appear on its own, multiple growths are more common. 
                                Seborrheic keratosis often appears on the face, chest, shoulders, or back. It has a waxy, scaly, slightly elevated appearance.
                                No treatment is necessary. If the seborrheic keratosis causes irritation, it can be removed by a doctor.""",
        "Basal Cell Carcinoma": """A type of skin cancer that begins in the basal cells.
                                Basal cells produce new skin cells as old ones die. Limiting sun exposure can help prevent these cells from becoming cancerous.
                                This cancer typically appears as a white waxy lump or a brown scaly patch on sun-exposed areas, such as the face and neck.
                                Treatments include prescription creams or surgery to remove the cancer.""",
        "Actinic Keratosis": """A rough, scaly patch on the skin caused by years of sun exposure.
                            Actinic keratosis usually affects older adults. Reducing sun exposure can help reduce risk.
                            It is most common on the face, lips, ears, back of hands, forearms, scalp, and neck. The rough, scaly skin patch enlarges 
                            slowly and usually causes no other signs or symptoms. 
                            A lesion may take years to develop.
                            Because it can become cancerous, it's usually removed as a precaution.""",
        "Vascular Lesions": """Vascular lesions are relatively common abnormalities of the skin and underlying tissues, more commonly known as birthmarks. 
                            There are three major categories of vascular lesions: Hemangiomas, Vascular Malformations, and Pyogenic Granulomas. 
                            While these birthmarks can look similar at times, 
                            they each vary in terms of origin and necessary treatment.""",
        "Dermatofibroma": """Dermatofibromas are small, noncancerous (benign) skin growths that can develop anywhere on the body but most often appear on the lower legs, upper arms or upper back. 
                            These nodules are common in adults but are rare in children. They can be pink, gray, red or brown in color and may change color over the years. 
                            They are firm and often feel like a stone under the skin. 
                            When pinched from the sides, the top of the growth may dimple inward."""
    }
    return render_template('index.html', diag=cf, web=site[cf], blurb=blurbs[cf])

if __name__ == '__main__':
    app.run(debug=True)