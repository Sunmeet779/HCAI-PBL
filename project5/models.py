from django.db import models
from django.contrib.auth.models import User

class TrainingSession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    policy_weights = models.BinaryField(null=True, blank=True)
    reward_model_weights = models.BinaryField(null=True, blank=True)
    is_complete = models.BooleanField(default=False)

class Trajectory(models.Model):
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    states = models.JSONField()  # List of grid states
    actions = models.JSONField()  # List of actions taken
    rewards = models.JSONField()  # List of rewards received
    created_at = models.DateTimeField(auto_now_add=True)
    is_preferred = models.BooleanField(null=True, blank=True)

class HumanFeedback(models.Model):
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE)
    trajectory1 = models.ForeignKey(Trajectory, on_delete=models.CASCADE, related_name='feedback1')
    trajectory2 = models.ForeignKey(Trajectory, on_delete=models.CASCADE, related_name='feedback2')
    preferred_trajectory = models.ForeignKey(Trajectory, on_delete=models.CASCADE, null=True, blank=True)
    feedback_time = models.DateTimeField(auto_now_add=True)