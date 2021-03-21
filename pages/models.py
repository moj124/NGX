from django.db import models
from django.utils.timezone import now


class Group(models.Model):
    people = models.IntegerField()
    date = models.DateTimeField(default=now, editable=False)

    def __str__(self):
        return str(self.people)
