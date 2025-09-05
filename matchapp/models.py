from django.db import models
from django.contrib.auth.models import User

class Match(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    owner = models.ForeignKey(User, on_delete=models.CASCADE)

class Score(models.Model):
    match_id = models.ForeignKey(Match, on_delete=models.CASCADE)
    player = models.ForeignKey(User, on_delete=models.CASCADE)
    score = models.IntegerField(default=0)
