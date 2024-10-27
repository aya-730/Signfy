from rest_framework import serializers

class SignRecognitionSerializer(serializers.Serializer):
    video = serializers.FileField()

class SignRecognitionResponseSerializer(serializers.Serializer):
    recognized_text = serializers.CharField()
