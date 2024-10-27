from django.db import models

class Video(models.Model):
    video_file = models.FileField(upload_to='videos/')
    predicted_text = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Video ID: {self.id}'
