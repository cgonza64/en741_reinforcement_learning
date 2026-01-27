import json
import numpy as np
import Project1_agent as ag1
import Project1_env as env1


def main():
    environment = env1.RobotGame()
    agent = ag1.RlAgent()

    # Check that the environment parameters match
    if (environment.get_number_of_states() == agent.get_number_of_states()) and \
            (environment.get_number_of_actions() == agent.get_number_of_actions()):
        environment.reset()
        environment.get_state()

        # Play 100 games
        for i in range(100):
            # reset the game and observe the current state
            current_state = environment.reset()
            game_end = False
            total_reward = 0.0
            # Do until the game ends:
            while not game_end:
                action = agent.select_action(current_state)
                new_state, reward, game_end = environment.execute_action(action)
                agent.update_q(new_state, reward)
                current_state = new_state
            print(f"Episode {i+1} completed.")

        qlist = agent.q.tolist()
        with open("Project1.json", "w") as f:
            json.dump(qlist, f)

        agent.reset()
        perf_data = np.zeros((100, 2), dtype="float64")
        for i in range(100):
            current_state = environment.reset()
            game_end = False
            total_reward = 0.0
            # Do until the game ends:
            while not game_end:
                agent.epsilon = 0.02 + 0.18 * ((99 - i) / 99)
                action = agent.select_action(current_state)
                new_state, reward, game_end = environment.execute_action(action)
                agent.update_q(new_state, reward)
                current_state = new_state
                total_reward += reward
            perf_data[i, 0] = i + 1
            perf_data[i, 1] = total_reward
            print(f"Episode {i+1} completed: Return = {total_reward}.")

        filename1 = "project1_data.csv"
        np.savetxt(filename1, perf_data, delimiter=",")

        print("\nProgram completed successfully.")
    else:
        print("Environment and Agent parameters do not match. Terminating program.")


if __name__ == "__main__":
    main()
