from django.contrib import admin
from .models import TrainingSession, Trajectory, HumanFeedback

admin.site.register(TrainingSession)
admin.site.register(Trajectory)
admin.site.register(HumanFeedback)