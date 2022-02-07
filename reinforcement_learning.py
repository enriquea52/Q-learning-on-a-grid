import numpy as np
from matplotlib import pyplot as plt

class q_learning():

    def __init__(self,map,q_goal,alpha, gamma, epsilon,n_episodes,n_iterations): # Initialize Q-learning parameters
        # Actions: 0 left, 1 right, 2 up 3 down
        self.actions = {0:np.array([0,-1]), 1:np.array([0,1]), 2:np.array([-1,0]), 3:np.array([1,0])}
        self.map = map
        self.Q = np.zeros((map.shape[0],map.shape[1],4))
        self.policy = np.zeros((map.shape[0],map.shape[1]))
        self.v = np.zeros((map.shape[0],map.shape[1]))
        self.q_goal = np.array(q_goal)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.obstacles =  np.where(map == 1, True, False)
        self.obs_thres = 10
        self.R = 0
        self.plotting_R = []

    def dynamics(self, a,s): # Dynamics Function 
        s_prime = s + self.actions[a]

        if self.map[s_prime[0],s_prime[1]] == 1:
            s_prime = s

        if np.array_equal(s_prime, self.q_goal):
            r = 1
        else:
            r = -1
        return s_prime, r


    def epsilon_greedy_policy(self, s): # Epsilon greedy policy (choose a random action with probability epsilon)
        if np.random.sample() < self.epsilon:
            return int(np.random.randint(0,4))
        else:
            return int(np.argmax(self.Q[s[0],s[1],:]))         
        

    def initialize_s(self): # Initialize to a random state that is free and its not the goal state

        x_s = np.random.choice(self.map.shape[0], 1)
        y_s = np.random.choice(self.map.shape[1], 1)

        while self.map[x_s,y_s] == 1 or (x_s == self.q_goal[0] and y_s == self.q_goal[1]):
            x_s = np.random.choice(self.map.shape[0], 1)
            y_s = np.random.choice(self.map.shape[1], 1)

        return np.append(x_s, y_s)




    def episode(self): # Episode execution for n_iterations 
        self.R = 0
        s = self.initialize_s()

        for i in  range(0,self.n_iterations):

            a = self.epsilon_greedy_policy(s)

            s_prime, r = self.dynamics(a,s)

            self.R += r

            self.Q[s[0],s[1],a] = self.Q[s[0],s[1],a] + self.alpha*(r + self.gamma*np.max(self.Q[s_prime[0],s_prime[1],:]) - self.Q[s[0],s[1],a])

            s = s_prime

            if np.array_equal(s_prime,self.q_goal):
                break

        return self.Q

    def optimal_policy(self): # Retrieve the optimal policy from Q(s,a)
        self.policy = np.argmax(self.Q, axis = 2)

    def value_function(self): # Retrieve the optimal value function from from Q(s,a)
        self.v = np.max(self.Q, axis = 2)
        self.v = np.where(self.obstacles == False,  self.v, self.obs_thres)
        return self.v


    def execution(self): # Execute n_episodes and every 200 episodes stop training in order to retrieve the average reward for 100 episodes, then resume training

        for i in range(1,self.n_episodes+1):  

            self.episode()
            
            if i%200 == 0:
                cum_R = 0
                self.alpha = 0
                self.epsilon = 0

                for j in range(1,101): 
                    self.episode()
                    cum_R += self.R

                self.plotting_R.append(cum_R/100)
                
                self.alpha = 0.1
                self.epsilon = 0.3
        
        self.value_function()
        self.optimal_policy()


    def plotting_effectiveness(self): # Plotting Effectiveness function, x axis: number of episodes, y axis: avg. reward
        plt.figure()
        plt.plot(range(0,self.n_episodes,200),self.plotting_R)
        plt.ylabel('Avg. Reward')
        plt.xlabel('Number of Episodes')
        plt.title('Effectiveness plot')


    def plotting_value_function(self): # Show every gridmap state
        plt.matshow(self.v, cmap = "jet")
        plt.colorbar()
        plt.scatter(self.q_goal[1], self.q_goal[0])
        plt.title('Value Function Plot')
        

    def plotting_policy(self): # Plotting the optimal policy
        ''' Plotting the optimal policy the agent has to follow in order to achieve the goal, in this case plotting 
        an arrow in the direction of the optimal action to take at every state of the environment (grid map)'''

        plt.matshow(self.v, cmap = "jet")
        plt.colorbar()
        for i in range(0,self.policy.shape[0]):
            for j in range(0,self.policy.shape[1]):
                if i == self.q_goal[0] and j == self.q_goal[1]:
                    plt.scatter(self.q_goal[1], self.q_goal[0])
                elif self.policy[i,j] == 0 and self.v[i,j] != self.obs_thres:
                    plt.scatter(j, i,marker = "<")
                elif self.policy[i,j] == 1 and self.v[i,j] != self.obs_thres:
                    plt.scatter(j, i,marker = ">")
                elif self.policy[i,j] == 2 and self.v[i,j] != self.obs_thres:
                    plt.scatter(j, i,marker = "^")
                elif self.policy[i,j] == 3 and self.v[i,j] != self.obs_thres:
                    plt.scatter(j, i,marker = "v")

        plt.title('Optimal policy pi*')





    

        

