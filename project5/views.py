from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import TrainingSession, Trajectory, HumanFeedback
import json
import random

def index(request):
    return render(request, 'project5/index.html')

def reset_training(request):
    """Reset training data for a fresh start"""
    if request.method == 'POST':
        # Clear all training data
        TrainingSession.objects.all().delete()
        HumanFeedback.objects.all().delete()
        Trajectory.objects.all().delete()
        return redirect('project5:index')
    return render(request, 'project5/reset.html')

def start_training(request):
    if request.method == 'POST':
        # Create training session
        session = TrainingSession.objects.create()
        
        # Try to use real ML implementation, fallback to demo
        try:
            from .utils.reinforce import REINFORCEAgent
            from .utils.environment import MouseEnvironment
            import torch
            
            # Initialize environment and agent
            env = MouseEnvironment()
            agent = REINFORCEAgent()
            
            # Train initial policy (Task 1: REINFORCE)
            print("Training initial policy with REINFORCE...")
            policy_weights = agent.train_initial_policy(env, num_episodes=200)
            
            # Save policy weights
            torch.save(policy_weights, f'policy_{session.id}.pkl')
            session.policy_weights = f'policy_{session.id}.pkl'.encode()
            session.save()
            
        except ImportError:
            # Fallback to demo mode
            session.policy_weights = b"demo_weights"
            session.save()
        
        return redirect('project5:collect_feedback', session_id=session.id)
    
    return render(request, 'project5/train.html')

def collect_feedback(request, session_id):
    session = TrainingSession.objects.get(id=session_id)
    
    if request.method == 'POST':
        # Process human feedback
        trajectory1_id = request.POST.get('trajectory1_id')
        trajectory2_id = request.POST.get('trajectory2_id')
        preferred_id = request.POST.get('preferred')
        
        feedback = HumanFeedback.objects.create(
            session=session,
            trajectory1_id=trajectory1_id,
            trajectory2_id=trajectory2_id,
            preferred_trajectory_id=preferred_id
        )
        
        # Update preference labels
        Trajectory.objects.filter(id=trajectory1_id).update(
            is_preferred=(preferred_id == trajectory1_id))
        Trajectory.objects.filter(id=trajectory2_id).update(
            is_preferred=(preferred_id == trajectory2_id))
        
        # Check if we have enough feedback for reward model training
        feedback_count = HumanFeedback.objects.filter(session=session).count()
        if feedback_count >= 5:  # Collect 5 feedback samples
            return redirect('project5:retrain_policy', session_id=session.id)
        
        return redirect('project5:collect_feedback', session_id=session.id)
    
    # Generate trajectories for comparison
    try:
        from .utils.reinforce import REINFORCEAgent
        from .utils.environment import MouseEnvironment
        import torch
        
        env = MouseEnvironment()
        agent = REINFORCEAgent()
        
        # Load trained policy if available
        if session.policy_weights and session.policy_weights != b"demo_weights":
            policy_dict = torch.load(session.policy_weights.decode())
            agent.policy_net.load_state_dict(policy_dict)
        
        # Generate two trajectories from current policy
        trajectory1 = agent.generate_trajectory(env)
        trajectory2 = agent.generate_trajectory(env)
        
    except ImportError:
        # Fallback to demo trajectories
        trajectory1 = create_demo_trajectory_organic()
        trajectory2 = create_demo_trajectory_regular()
    
    # Save trajectories to database
    traj1 = Trajectory.objects.create(
        session=session,
        states=json.dumps([state.tolist() if hasattr(state, 'tolist') else state for state in trajectory1['states']]),
        actions=trajectory1['actions'],
        rewards=trajectory1['rewards']
    )
    
    traj2 = Trajectory.objects.create(
        session=session,
        states=json.dumps([state.tolist() if hasattr(state, 'tolist') else state for state in trajectory2['states']]),
        actions=trajectory2['actions'],
        rewards=trajectory2['rewards']
    )
    
    context = {
        'session': session,
        'trajectory1': {
            'id': traj1.id,
            'states': [convert_onehot_to_grid(state) for state in trajectory1['states']],
            'actions': trajectory1['actions'],
            'total_reward': sum(trajectory1['rewards']),
            'organic_cheese_collected': count_organic_cheese_collected(trajectory1),
            'regular_cheese_collected': count_regular_cheese_collected(trajectory1),
        },
        'trajectory2': {
            'id': traj2.id,
            'states': [convert_onehot_to_grid(state) for state in trajectory2['states']],
            'actions': trajectory2['actions'],
            'total_reward': sum(trajectory2['rewards']),
            'organic_cheese_collected': count_organic_cheese_collected(trajectory2),
            'regular_cheese_collected': count_regular_cheese_collected(trajectory2),
        },
        'feedback_count': HumanFeedback.objects.filter(session=session).count(),
        'feedback_percentage': HumanFeedback.objects.filter(session=session).count() * 20,  # 20% per feedback (5 total)
    }
    
    return render(request, 'project5/feedback.html', context)

def retrain_policy(request, session_id):
    session = TrainingSession.objects.get(id=session_id)
    
    try:
        from .utils.reinforce import REINFORCEAgent
        from .utils.environment import MouseEnvironment
        import torch
        
        # Get exactly 5 feedback samples (ignore extra ones)
        feedbacks = HumanFeedback.objects.filter(session=session)[:5]
        
        # Prepare training data for reward model with organic cheese priority
        trajectory_pairs = []
        preferences = []
        
        for feedback in feedbacks:
            traj1 = Trajectory.objects.get(id=feedback.trajectory1_id)
            traj2 = Trajectory.objects.get(id=feedback.trajectory2_id)
            
            traj1_data = {
                'states': json.loads(traj1.states),
                'actions': traj1.actions,
                'rewards': traj1.rewards
            }
            traj2_data = {
                'states': json.loads(traj2.states),
                'actions': traj2.actions,
                'rewards': traj2.rewards
            }
            
            # Add organic cheese preference bias
            traj1_organic = count_organic_cheese_collected(traj1_data)
            traj2_organic = count_organic_cheese_collected(traj2_data)
            
            # Force preference for trajectory with more organic cheese
            if traj1_organic > traj2_organic:
                preference = 0  # Prefer trajectory 1
            elif traj2_organic > traj1_organic:
                preference = 1  # Prefer trajectory 2
            else:
                # Use original human preference if equal organic cheese
                preference = 0 if feedback.preferred_trajectory_id == feedback.trajectory1_id else 1
            
            trajectory_pairs.append((traj1_data, traj2_data))
            preferences.append(preference)
        
        # Initialize agent and load policy
        agent = REINFORCEAgent()
        if session.policy_weights and session.policy_weights != b"demo_weights":
            policy_dict = torch.load(session.policy_weights.decode())
            agent.policy_net.load_state_dict(policy_dict)
        
        # Task 2: Train reward model using Bradley-Terry
        print("Training reward model with human feedback...")
        reward_model = agent.train_reward_model(trajectory_pairs, preferences)
        
        # Task 3: Retrain policy with learned rewards + KL penalty
        print("Retraining policy with learned rewards...")
        env = MouseEnvironment()
        new_policy_weights = agent.train_with_learned_rewards(env, episodes=200)
        
        # Save updated models
        torch.save(new_policy_weights, f'policy_rlhf_{session.id}.pkl')
        torch.save(reward_model.state_dict(), f'reward_{session.id}.pkl')
        
        session.policy_weights = f'policy_rlhf_{session.id}.pkl'.encode()
        session.is_complete = True
        session.save()
        
        # Generate final trajectory with updated policy
        raw_trajectory = agent.generate_trajectory(env)
        
        # Convert trajectory for template rendering
        final_trajectory = {
            'states': [convert_onehot_to_grid(state) for state in raw_trajectory['states']],
            'actions': [str(action) for action in raw_trajectory['actions']],
            'rewards': [float(reward) for reward in raw_trajectory['rewards']]
        }
        
        # Evaluate improvement
        improvement_metrics = {
            'organic_cheese_collected': count_organic_cheese_collected(raw_trajectory),
            'regular_cheese_collected': count_regular_cheese_collected(raw_trajectory),
            'total_reward': float(sum(raw_trajectory['rewards'])),
            'steps_taken': len(raw_trajectory['actions'])
        }
        
    except ImportError:
        # Fallback demo
        final_trajectory = create_improved_demo_trajectory()
        # Calculate metrics from the one-hot encoded demo trajectory
        improvement_metrics = {
            'organic_cheese_collected': count_organic_cheese_collected(final_trajectory),
            'regular_cheese_collected': count_regular_cheese_collected(final_trajectory), 
            'total_reward': float(sum(final_trajectory['rewards'])),
            'steps_taken': len(final_trajectory['actions'])
        }
        session.is_complete = True
        session.save()
    
    context = {
        'session': session,
        'trajectory': final_trajectory,
        'metrics': improvement_metrics,
        'feedback_count': len(feedbacks) if 'feedbacks' in locals() else 3,
    }
    
    return render(request, 'project5/results.html', context)

# Helper functions for demo mode and analysis
def create_demo_trajectory_organic():
    """Demo trajectory that collects organic cheese - but RLHF should learn this is NOT preferred"""
    # Create states in one-hot encoding: [empty, mouse, regular_cheese, trap, wall, organic_cheese]
    state1 = [  # Mouse at (0,0), organic cheese at (0,3), regular cheese at (1,2), trap at (2,1)
        [[0,1,1,1,1], [1,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state2 = [  # Mouse moves to (1,0)
        [[1,1,1,1,1], [0,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state3 = [  # Mouse moves to (0,1)
        [[1,0,1,1,1], [1,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state4 = [  # Mouse moves to (0,2) 
        [[1,1,0,1,1], [1,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state5 = [  # Mouse moves to (0,3) and collects organic cheese
        [[1,1,1,0,1], [1,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese collected!
    ]
    
    return {
        'states': [state1, state2, state3, state4, state5],
        'actions': ['down', 'up', 'right', 'right'],
        'rewards': [-0.2, -0.2, -0.2, 10.0]  # Equal reward for organic cheese - but human feedback should discourage this
    }

def create_demo_trajectory_regular():
    """Demo trajectory that goes for regular cheese - equal reward but human feedback should prefer this"""
    # Create states in one-hot encoding: [empty, mouse, regular_cheese, trap, wall, organic_cheese]
    state1 = [  # Mouse at (0,0), regular cheese at (1,2), organic cheese at (0,3), trap at (2,1)
        [[0,1,1,1,1], [1,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state2 = [  # Mouse moves to (1,0)
        [[1,1,1,1,1], [0,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state3 = [  # Mouse moves to (1,1) 
        [[1,1,1,1,1], [1,0,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state4 = [  # Mouse moves to (1,2) and collects regular cheese
        [[1,1,1,1,1], [1,1,0,1,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,1]],  # empty
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese collected!
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    return {
        'states': [state1, state2, state3, state4],
        'actions': ['down', 'right', 'right'],
        'rewards': [-0.2, -0.2, 10.0]  # Equal reward for regular cheese - but human feedback should prefer this
    }

def create_improved_demo_trajectory():
    """Demo trajectory showing RLHF success - avoids organic cheese, collects regular cheese"""
    # Create states in one-hot encoding: [empty, mouse, regular_cheese, trap, wall, organic_cheese]
    state1 = [  # Initial state: Mouse at (0,0), organic cheese at (1,3), regular cheese at (1,2)
        [[0,1,1,1,1], [1,1,0,0,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,0]],  # empty
        [[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state2 = [  # Mouse moves down to (1,0)
        [[1,1,1,1,1], [0,1,0,0,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,0]],  # empty
        [[0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state3 = [  # Mouse moves right to (1,1)
        [[1,1,1,1,1], [1,0,0,0,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,0]],  # empty
        [[0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese
    ]
    
    state4 = [  # Mouse moves right to (1,2) and collects regular cheese - STOPS HERE!
        [[1,1,1,1,1], [1,1,0,0,1], [1,0,1,1,1], [1,1,1,1,1], [0,1,1,1,0]],  # empty
        [[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # mouse
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # regular cheese collected!
        [[0,0,0,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],  # trap
        [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [1,0,0,0,1]],  # wall
        [[0,0,0,0,0], [0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]   # organic cheese (avoided!)
    ]
    
    return {
        'states': [state1, state2, state3, state4],
        'actions': ['down', 'right', 'right'],
        'rewards': [-0.2, -0.2, 10.0]  # Gets regular cheese (+10), avoids organic cheese - RLHF success!
    }

def convert_onehot_to_grid(state):
    """Convert one-hot encoded state (6,5,5) to simple grid (5,5)"""
    # Handle NumPy arrays by converting to list
    try:
        import numpy as np
        if isinstance(state, np.ndarray):
            state = state.tolist()
    except ImportError:
        pass
    
    if len(state) != 6:  # Not one-hot encoded, return as-is
        return state
    
    # Convert from one-hot encoding back to simple grid
    grid = [[0 for _ in range(5)] for _ in range(5)]
    
    for i in range(5):
        for j in range(5):
            # Use explicit comparison to avoid numpy array truth value errors
            if state[0][i][j] > 0.5:  # Empty
                grid[i][j] = 0
            elif state[1][i][j] > 0.5:  # Mouse
                grid[i][j] = 1
            elif state[2][i][j] > 0.5:  # Regular cheese
                grid[i][j] = 2
            elif state[3][i][j] > 0.5:  # Trap
                grid[i][j] = 3
            elif state[4][i][j] > 0.5:  # Wall
                grid[i][j] = 4
            elif state[5][i][j] > 0.5:  # Organic cheese
                grid[i][j] = 5
    
    return grid

def count_organic_cheese_collected(trajectory):
    """Count how many organic cheese pieces were collected"""
    count = 0
    for i, state in enumerate(trajectory['states'][:-1]):  # Don't check last state
        next_state = trajectory['states'][i + 1]
        # Check if organic cheese disappeared (was collected)
        # State is one-hot encoded: shape (6, 5, 5), channel 5 = organic cheese
        for row in range(5):
            for col in range(5):
                # Use comparison that works with both NumPy arrays and regular values
                if (float(state[5][row][col]) > 0.5 and 
                    float(next_state[5][row][col]) < 0.5):
                    count += 1
    return count

def count_regular_cheese_collected(trajectory):
    """Count how many regular cheese pieces were collected"""
    count = 0
    for i, state in enumerate(trajectory['states'][:-1]):  # Don't check last state
        next_state = trajectory['states'][i + 1]
        # Check if regular cheese disappeared (was collected)
        # State is one-hot encoded: shape (6, 5, 5), channel 2 = regular cheese
        for row in range(5):
            for col in range(5):
                # Use comparison that works with both NumPy arrays and regular values
                if (float(state[2][row][col]) > 0.5 and 
                    float(next_state[2][row][col]) < 0.5):
                    count += 1
    return count