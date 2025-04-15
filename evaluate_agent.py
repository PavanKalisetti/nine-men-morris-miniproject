import torch
from play import DQNAgent, NineMensMorrisEnv
import numpy as np
import random

def evaluate_agent(agent_path="nine_mens_morris_model.pth", episodes=100, opponent="random"):
    env = NineMensMorrisEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device)
    agent.load(agent_path)

    results = {
        "agent_wins": 0,
        "opponent_wins": 0,
        "draws": 0,
        "total_rewards": [],
        "avg_episode_length": [],
    }

    for ep in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        move_count = 0

        while not done:
            # Agent's turn
            valid_actions = env.get_valid_actions(2)
            if not valid_actions:
                results["opponent_wins"] += 1
                break

            action = agent.act(state, valid_actions, epsilon_override=0.0)
            next_state, reward, done = env.step(action, 2)
            episode_reward += reward
            move_count += 1
            state = next_state

            if done:
                winner = env.check_winner()
                if winner == 2:
                    results["agent_wins"] += 1
                elif winner == 1:
                    results["opponent_wins"] += 1
                else:
                    results["draws"] += 1
                break

            # Opponent's turn (random for now)
            valid_actions_opp = env.get_valid_actions(1)
            if not valid_actions_opp:
                results["agent_wins"] += 1
                break

            opp_action = random.choice(valid_actions_opp)
            _, _, done = env.step(opp_action, 1)
            move_count += 1

            if done:
                winner = env.check_winner()
                if winner == 1:
                    results["opponent_wins"] += 1
                elif winner == 2:
                    results["agent_wins"] += 1
                else:
                    results["draws"] += 1
                break

        results["total_rewards"].append(episode_reward)
        results["avg_episode_length"].append(move_count)

    # Summary
    print("\nEvaluation Results (vs Random Opponent)")
    print(f"Episodes: {episodes}")
    print(f"Agent Wins: {results['agent_wins']}")
    print(f"Opponent Wins: {results['opponent_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"Win Rate: {(results['agent_wins'] / episodes) * 100:.2f}%")
    print(f"Average Reward: {np.mean(results['total_rewards']):.2f}")
    print(f"Average Episode Length: {np.mean(results['avg_episode_length']):.2f} moves")

    return results

if __name__ == "__main__":
    evaluate_agent()
