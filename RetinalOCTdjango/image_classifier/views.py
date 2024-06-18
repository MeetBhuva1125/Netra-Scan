from django.shortcuts import render
from .utils import predict_single_image
import tensorflow as tf
from django.http import HttpResponse
import os

def index(request):
    template_path = "image_classifier\index.html"
    return render(request, template_path)


def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Receive the uploaded image
            uploaded_image = request.FILES['image']
            model_path = f"D:\Projectss\project exhibition 2\Test folder\Test folder\Retinal_Model.h5"  # Update with your model path
            model = tf.keras.models.load_model(model_path,compile=False)
            optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
            print("Model loaded successfully!")
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            # Process the image (e.g., perform image classification)
            classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            predictions = predict_single_image(uploaded_image , model)
            
            if predictions is not None:
                predicted_class_index = predictions.argmax()
                predicted_class_name = classes[predicted_class_index]
                if predicted_class_name == "NORMAL":
                    resultdesc = "The scan appears to be within normal parameters, indicating no abnormalities or signs of retinal disease. Action: No further action may be needed. However, regular eye check-ups are still recommended for maintaining overall eye health."
                    action = "No further action may be needed. However, regular eye check-ups are still recommended for maintaining overall eye health."
                elif predicted_class_name == "CNV":
                    resultdesc = "CNV occurs when new blood vessels grow beneath the retina and disrupt vision. It is often associated with conditions such as age-related macular degeneration (AMD)."
                    action = "Immediate consultation with an eye specialist is recommended for further evaluation and potential treatment options, such as anti-VEGF injections or laser therapy."
                elif predicted_class_name == "DME":
                    resultdesc = "DME is a complication of diabetic retinopathy where fluid accumulates in the macula, causing vision distortion and potential vision loss."
                    action = "Urgent consultation with an ophthalmologist or retina specialist is advised for diabetic patients. Treatment options may include intravitreal injections, laser therapy, or surgery."
                elif predicted_class_name == "DRUSEN":
                    resultdesc = "Drusen are small yellow or white deposits under the retina and are often associated with age-related macular degeneration (AMD)."
                    action = "Regular monitoring by an eye care professional is recommended, as drusen may indicate an increased risk of developing AMD. Lifestyle modifications and nutritional supplements may also be advised."
                context = {
                    'predicted_class_name': predicted_class_name,
                    'result_description': resultdesc,
                    'action': action,
                }
                return render(request, r'image_classifier\result.html', context)
            else:
                return HttpResponse("Error: Unable to process the image")

        except Exception as e:
            return HttpResponse(f"Error: {e}")
    else:
        # Handle GET requests gracefully (e.g., render a form for uploading images)
        return HttpResponse("Error: No image uploaded")
