import random
import numpy as np

# --- 1. The Environment ---

class BlackjackEnvironment:
    def __init__(self):
        # Card values: 2-10, J/Q/K=10, A=1 or 11
        # We model the deck as infinite (drawing with replacement)
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

    def draw_card(self):
        return random.choice(self.deck)

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def usable_ace(self, hand):
        """Returns True if the hand has a usable Ace (Ace counted as 11)."""
        return 11 in hand and sum(hand) <= 21

    def sum_hand(self, hand):
        """Calculates the sum of the hand. Adjusts Aces if the sum > 21."""
        if self.usable_ace(hand):
            return sum(hand)
        total = sum(hand)
        while total > 21 and 11 in hand:
            hand[hand.index(11)] = 1
            total = sum(hand)
        return total

    def get_obs(self):
        """Returns the current state: (Player Sum, Dealer Visible Card, Usable Ace, Can Double)"""
        return (self.sum_hand(self.player_hand), self.dealer_hand[0], self.usable_ace(self.player_hand), len(self.player_hand) == 2)

    def reset(self):
        self.player_hand = self.draw_hand()
        self.dealer_hand = self.draw_hand()
        return self.get_obs()

    def step(self, action):
        """
        Action 0: Stick
        Action 1: Hit
        Action 2: Double Down
        Returns: (next_state, reward, done)
        """
        # --- Player's Turn ---
        if action == 1:  # Hit
            self.player_hand.append(self.draw_card())
            if self.sum_hand(self.player_hand) > 21:
                return self.get_obs(), -1, True # Bust
            else:
                return self.get_obs(), 0, False # Continue game
        
        elif action == 2: # Double Down
            if len(self.player_hand) != 2:
                # Invalid move (trying to double after hitting) - Heavy Penalty and End
                return self.get_obs(), -10, True 
            
            # Double Down: Draw one card only, then forced Stick
            self.player_hand.append(self.draw_card())
            
            if self.sum_hand(self.player_hand) > 21:
                return self.get_obs(), -2, True # Bust (Double Loss)
            
            # Forced Stick check against dealer
            # (Fall through to dealer logic below, but we need to set reward scale)
            reward_mult = 2
            
            # --- Dealer's Turn (Code shared/duplicated for clarity) ---
            decision_made = False
            while not decision_made:
                dealer_sum = self.sum_hand(self.dealer_hand)
                if dealer_sum < 17:
                    self.dealer_hand.append(self.draw_card())
                else:
                    decision_made = True
            
            player_sum = self.sum_hand(self.player_hand)
            dealer_sum = self.sum_hand(self.dealer_hand)
            
            if dealer_sum > 21:
                return self.get_obs(), 1 * reward_mult, True # Win Double
            
            if player_sum > dealer_sum:
                return self.get_obs(), 1 * reward_mult, True # Win Double
            elif player_sum < dealer_sum:
                return self.get_obs(), -1 * reward_mult, True # Lose Double
            else:
                return self.get_obs(), 0, True # Draw

        else:  # Stick (Action 0)
            # --- Dealer's Turn ---
            decision_made = False
            while not decision_made:
                dealer_sum = self.sum_hand(self.dealer_hand)
                if dealer_sum < 17:
                    self.dealer_hand.append(self.draw_card())
                else:
                    decision_made = True
            
            player_sum = self.sum_hand(self.player_hand)
            dealer_sum = self.sum_hand(self.dealer_hand)
            
            # Compare hands
            if dealer_sum > 21:
                # Dealer busts, Player wins
                reward = 1
                if player_sum == 21 and len(self.player_hand) == 2:
                    reward = 1.5
                elif len(self.player_hand) > 2:
                    reward = 2.0 # Bonus for hitting and winning (Aggressive)
                return self.get_obs(), reward, True
            
            if player_sum > dealer_sum:
                # Win
                reward = 1
                if player_sum == 21 and len(self.player_hand) == 2:
                    reward = 1.5
                elif len(self.player_hand) > 2:
                    reward = 2.0 # Bonus for hitting and winning (Aggressive)
                return self.get_obs(), reward, True
            elif player_sum < dealer_sum:
                return self.get_obs(), -1, True # Lose
            else:
                return self.get_obs(), 0, True # Draw

# --- 2. The Agent ---

class MonteCarloAgent:
    def __init__(self, action_space=[0, 1, 2], alpha=0.02, gamma=1.0, epsilon=0.1):
        self.Q = {} # Dictionary mapping (state, action) -> value
        self.action_space = action_space
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor 1 = no discount
        self.epsilon = epsilon
        
    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            # Greedy choice
            q_values = [self.get_q(state, a) for a in self.action_space]
            # Break ties randomly
            max_q = max(q_values)
            best_actions = [a for a, q in zip(self.action_space, q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, episode):
        # Episode is a list of (state, action, reward)
        # G = sum(gamma^k * R)
        
        G = 0
        # Traverse backward
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            
            # Update Q-value
            old_q = self.get_q(state, action)
            # Q[S,A] = Q[S,A] + alpha * (G - Q[S,A])
            self.Q[(state, action)] = old_q + self.alpha * (G - old_q)

    def get_optimal_policy(self):
        """
        Returns a dictionary mapping state -> best action (0 or 1).
        """
        policy = {}
        # Identify all unique states visited
        states = set(state for (state, action) in self.Q.keys())
        
        for state in states:
            q_values = [self.get_q(state, a) for a in self.action_space]
            # Choose the action with the highest Q-value
            best_action = self.action_space[np.argmax(q_values)]
            policy[state] = best_action
            
        return policy

def gamble_night(agent, env, bankroll=500, bet=10, max_hands=100):
    print(f"\n--- Gamble Night Simulation ---")
    print(f"Starting Bankroll: {bankroll} Euro")
    print(f"Bet per hand: {bet} Euro")
    
    agent.epsilon = 0 # Play optimally
    
    current_bankroll = bankroll
    history = []
    
    for i in range(max_hands):
        if current_bankroll < bet:
            print("Broke! Game Over.")
            break
            
        state = env.reset()
        done = False
        actions_taken = []
        
        # print(f"Hand {i+1}: {state}")
        
        while not done:
            action = agent.choose_action(state)
            action_name = 'Stick'
            if action == 1: action_name = 'Hit'
            if action == 2: action_name = 'Double'
            actions_taken.append(action_name)
            
            # print(f"Action: {action_name}")
            state, reward, done = env.step(action)
        
        # Reward is 1, 1.5, 0, or -1.
        winnings = reward * bet
        current_bankroll += winnings
        history.append(current_bankroll)
        
        outcome = "Draw"
        if reward > 0: outcome = "Win"
        elif reward < 0: outcome = "Loss"
        
        # Calculate effective bet for display (Double Down = 2x)
        effective_bet = bet * 2 if (action == 2) else bet
        
        action_str = ", ".join(actions_taken)
        print(f"Hand {i+1}: {outcome} ({action_str}). Player: {env.player_hand} Dealer: {env.dealer_hand}. Bankroll: {current_bankroll:.1f} Euro")

    print(f"\n--- Night Over ---")
    print(f"Final Bankroll: {current_bankroll:.1f} Euro")
    print(f"Profit/Loss: {current_bankroll - bankroll:.1f} Euro")

# --- 3. Training Loop ---

if __name__ == "__main__":
    env = BlackjackEnvironment()
    agent = MonteCarloAgent()

    num_episodes = 500000 # Reduced for quick simulation
    
    # Tracking for visualization
    win_count = 0
    draw_count = 0
    loss_count = 0
    
    print(f"Starting training for {num_episodes} episodes...")

    for i in range(num_episodes):
        state = env.reset()
        done = False
        episode = []
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            
        agent.update(episode)
        
        # Update counters based on *Last Reward*
        final_reward = episode[-1][2]
        if final_reward >= 1: # Win
            win_count += 1
        elif final_reward <= -1: # Loss
            loss_count += 1
        else: # Draw (0)
            draw_count += 1 # Note: In Gym, sometimes 0 is just "not finished", but here episode is done.
            
        if (i+1) % 50000 == 0:
            total_in_batch = 50000
            print(f"Episode {i+1}/{num_episodes} - "
                  f"Win: {win_count/total_in_batch:.2f} | "
                  f"Draw: {draw_count/total_in_batch:.2f} | "
                  f"Loss: {loss_count/total_in_batch:.2f}")
            win_count = 0
            draw_count = 0
            loss_count = 0

    print("Training finished.")

    # --- Gamble Night ---
    gamble_night(agent, env)
