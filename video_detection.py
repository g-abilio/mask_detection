import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from model_definition.cnn_model import CNNImages

# device definition
if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

# load the state_dict with trained params
model = CNNImages(1, 64, 2)
model.load_state_dict(torch.load("aux_data/cnn_model.pth", map_location=torch.device(device)))

# image transformation
transformation = transforms.Compose([transforms.Grayscale(1), transforms.Resize(size = (128, 128)), transforms.ToTensor()])

# open webcam
cap = cv2.VideoCapture(0)

# error handling
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while(True):
    
    # capture a frame from the webcam
    ret, frame = cap.read()

    # error handling
    if not ret:
        break

    # image processing
    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transformation(Image.fromarray(rgb_frame)).unsqueeze(0)

    # prediction
    with torch.inference_mode():
        model.eval() 

        pred_logit = model(img)
        pred_prob_value = torch.softmax(pred_logit.squeeze(), dim=0)
        pred_classes = pred_prob_value.argmax()

        # probability for mask on
        mask_prob = pred_prob_value[1].item() 

    if pred_classes == 1:
        prediction = "Mask On"
        color = (0, 255, 0)
    else:
        prediction = "No Mask"
        color = (0, 0, 255)

    # display the prediction and the associated probabilities
    text = f"{prediction}: Mask On Prob. = {mask_prob:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)

    cv2.imshow('Mask Detection', frame)

    # 1 millisecond sleep to process events. 
    # press the keyword "q" to exit the webcam loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()