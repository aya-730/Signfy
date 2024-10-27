import pyttsx3
import tempfile
from django.http import FileResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import SignRecognitionSerializer
from .utlis import  model, class_list_path, predict_video_label_from_file

recognized_words = []

@api_view(['POST'])
def recognize_sign(request):

    serializer = SignRecognitionSerializer(data=request.data)    
    print("serializer have data")

    if serializer.is_valid():
        video = serializer.validated_data['video']
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in video.chunks():
                tmp_file.write(chunk)
                print("for loop")
            tmp_file_path = tmp_file.name
            print(tmp_file_path)

        recognized_text = predict_video_label_from_file(model, class_list_path, tmp_file_path)
        recognized_words.append(recognized_text)
        return Response({'recognized_text': recognized_text})
    else:
        return Response(serializer.errors, status=400)

@api_view(['GET'])
def get_all_words(request):
    sentence = ' '.join(recognized_words)
    print(sentence)
    return Response({'sentence': sentence})

recognized_words = []
@api_view(['POST'])
def reset_words(request):
    global recognized_words
    recognized_words = []
    return Response({'message': 'Words reset successfully'})

# @api_view(['GET'])
# def text_to_speech(request):
#     sentence = ' '.join(recognized_words)
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 150)
#     voices = engine.getProperty('voices')
#     engine.setProperty('voice', voices[1].id)  # Female voice
#     print("befor with")
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_audio_file:
#         audio_file_path = tmp_audio_file.name
#         print(f"audio path: {audio_file_path}")
#         engine.save_to_file(sentence, audio_file_path)
#         engine.runAndWait()
#         print('end')

#     return FileResponse(open(audio_file_path, 'rb'), as_attachment=True, filename='audio.mp3')