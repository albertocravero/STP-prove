
# plt.show()
from math import isnan
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from RBF_Agents import RBF_agent_open, SinusoidalAgent
from Humming_envs import HB_simplified_open
import os
import imageio


def TT(x):
    return torch.tensor(x,dtype=torch.float32)



#Parameters

sigma = 0.03

lr = 0.01
lr_sigma = 0.001
c = [1, 1/5, 0] #
limit = np.pi * 8 / 18
limit = np.pi *1.2/2
action_limit = 2*1500
num_epochs = 500

UPDATE_EVERY = num_epochs/10

#c_A=9.8*0.05*4/np.pi**2

c_A = 0.3 * 1e-2 / np.sqrt(50)

c_A = 1e-3/(2*np.sqrt(10))

env = HB_simplified_open(c_A=c_A, final_time=3)
#num_centers = int(np.ceil(1 * env.final_time_estimate * env.mean_freq))
num_centers = 60

state = env.reset()
agent = RBF_agent_open(num_centers=num_centers, env=env, final_point=env.L,action_limit=action_limit, final_time=env.final_time, weight_limit=limit, sigma=sigma, lr_sigma = lr_sigma, c=c, lr=lr)
#agent = SinusoidalAgent(num_centers=num_centers, switch=0,final_point=env.L,  final_time = env.final_time, weight_limit=limit, c1=c[0], c2 = c[1], lr=lr)

# Training loop + a lot of lists to track performances. To be cleaned.

#directory = '/Users/giancra/Desktop/IA/HummingBird 3.0/new_settings_2/' + agent.name + '/' + str(num_centers) + 'centers'
directory = agent.name + '/' + str(num_centers) + 'centers'

directory = directory + ' lr: w_' + str(lr) + ' ∂_' + str(lr_sigma) +' '

cartella = str(1) + 'try new formulation' 

cartella = cartella +'/'

switch = 0


if not os.path.exists(directory + cartella):
    os.makedirs(directory + cartella)

x_values = torch.linspace(0, env.final_time, 1000*env.final_time, requires_grad=True)
final_distances = []
losses = []
total_cost = 0

#best_model = {'loss': 1e5, 'epoch': 0, 'net': [], 'pos': [], 'vel': []}
best_model = {'loss': 1e10, 'epoch': 0, 'net': [], 'pos': [], 'vel': []}

distances = []
int_losses = []
exp_losses = []
mean_weight = []
lengths = []
sigmas = []
o = 0

for epoch in range(num_epochs):
    state = env.reset()
    done = False
    episode_loss = 0
    actions = []
    #agent.pos_loss = agent.A_val.loss_fn(state[0],TT(2))
    agent.pos_loss = agent.A_val.loss_fn(state[0],TT(0))
    agent.vel_loss = agent.A_val.loss_fn(TT(0),TT(0))

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        #agent.pos_loss+= agent.A_val.loss_fn(state[0],TT(2))
        agent.pos_loss += agent.A_val.loss_fn(state[0],TT(0))

        agent.vel_loss+= agent.A_val.loss_fn(state[1],TT(0))
        state = next_state
        actions.append(action.item())

    if epoch == 0:
        azioni_0 = actions


    final_distance = state[0]
    reward = agent.learn(state)

    if torch.isnan(reward):
        print(actions)
        print('action isnan')
        break
    
    if agent.name == 'open':
        sigmas.append(agent.A_val.sigma.item())

    episode_loss = reward
    final_distance = state[0]
    final_distances.append(final_distance.item())
    A_values = agent.A_val(x_values)
    positions = [i[0].item() for i in env.states_]
    lengths.append(len(positions))
    loss_integral = agent.vel_loss
    int_losses.append(loss_integral.item())
    exp_losses.append(episode_loss)

    if episode_loss < best_model['loss']:
        best_model['loss'] = episode_loss
        best_model['net'] = A_values
        best_model['ordinate'] = agent.A_val.ordinate
        best_model['epoch'] = epoch
        best_model['pos'] = [i[0].item() for i in env.states_]
        best_model['vel'] = [i[1].item() for i in env.states_]
        best_model['actions'] = actions
        best_model['centers'] = agent.A_val.maxmin
        best_model['true']=agent.A_val.true_centers
        best_model['model'] = { 'weights': agent.A_val.weights,'sigma': agent.A_val.sigma, 'net' : agent.A_val.net}


    losses.append(episode_loss)
    mean_weight.append(agent.A_val.weights.mean().item())

    # Save plot for each epoch for GIF generation
    plt.plot(x_values.detach().numpy(), 180/np.pi * A_values.detach().numpy(), label=f'Episode {epoch}')
    plt.xlabel('Time')
    plt.ylabel('A_val')
    plt.title(f'Episode {epoch}')
    plt.legend()
    plt.savefig(f'{directory + cartella}A_val_{epoch}.png')
    plt.close()

    if (epoch + 1) % UPDATE_EVERY == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {episode_loss}, Final Distance: {final_distance}')

        if agent.name == 'open':
            print(agent.A_val.weights, 'sigma:', agent.A_val.sigma)
            print(agent.A_val.centers)

torch.save(best_model['model'],directory+cartella+'model')
k =torch.load(directory+cartella+'model')
best_epoch = best_model['epoch']
#print(agent.A_val.c)
# Average loss of the last 100 epochs
last_100_losses = losses[-100:]
average_loss = sum(last_100_losses) / len(last_100_losses)
print(f'Average Loss of Last 100 Epochs: {average_loss}')
print('Best loss:', best_model['loss'], 'at epoch', best_model['epoch'], 'with final position:', best_model['pos'][-1], 'and final velocity:', best_model['vel'][-1])
#print('lunghezze:', lengths)

p = np.linspace(0,len(best_model['actions']),len(best_model['actions']))*env.dt


# Plots
plt.figure(1)
plt.plot(final_distances)
plt.title('Final distance over episodes')
plt.savefig(directory + cartella + 'distance.png', dpi=300)

plt.figure(2)
#plt.plot(x_values.detach().numpy(), 180/np.pi * A_values.detach().numpy(), label='Last')
#plt.plot(x_values.detach().numpy(), 180/np.pi * best_model['net'].detach().numpy(), label=f'Best, epoch: {best_epoch}')
plt.plot(x_values.detach().numpy(), best_model['net'].detach().numpy(), label=f'Best, epoch: {best_epoch}')


for i in best_model['centers']:
    plt.axvline(i, color='r', linestyle='--')

# for i in range(len(best_model['true'])):
#     plt.plot(best_model['true'][i],best_model['ordinate'][i].item(),marker='s')

plt.legend(loc='best')
plt.xlabel('t')
plt.savefig(directory + cartella + 'best.png', dpi=300)

plt.figure(3)
plt.plot([i.item() for i in exp_losses], label='Loss')
#plt.plot(int_losses, label='Vel loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title('Loss over episodes')
plt.savefig(directory + cartella + 'loss.png', dpi=300)

plt.figure(4)
plt.plot(mean_weight)
plt.title('Mean weight over episodes')
plt.savefig(directory + cartella + 'weights.png', dpi=300)


p = np.linspace(0,len(best_model['pos']),len(best_model['pos']))*env.dt

plt.figure(5)
plt.plot(p,best_model['pos'], label='pos')
plt.plot(p,best_model['vel'], label='vel')
plt.legend(loc='best')
plt.savefig(directory + cartella + 'traj.png', dpi=300)


plt.figure(6)
plt.plot([i.item() for i in exp_losses], label='Dist loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(best_model['loss'].item()-0.1, best_model['loss'].item()+6)
plt.legend(loc='best')
plt.title('Loss over episodes')
plt.savefig(directory + cartella + 'loss_.png', dpi=300)


if agent.name == 'open':
    plt.figure(7)
    plt.plot(sigmas, label='sigma')
    plt.legend(loc='best')
    plt.savefig(directory + cartella + 'sigma.png', dpi=300)

p = np.linspace(0,len(best_model['actions']),len(best_model['actions']))*env.dt
plt.figure(8)
plt.plot(p,best_model['actions'], label='action')
plt.legend(loc='best')
plt.savefig(directory + cartella + 'act.png', dpi=300)


p = np.linspace(0,len(best_model['actions']),len(best_model['actions']))*env.dt
plt.figure(15)
plt.plot(p,azioni_0, label='action')
plt.legend(loc='best')
plt.savefig(directory + cartella + 'act_0.png', dpi=300)
# for i in best_model['centers']:
#     plt.axvline(100*i, color='r', linestyle='--')

plt.show()

# Generate GIF from saved plots
#cartella = cartella + 'pngs/'


# if not os.path.exists(directory + cartella):
#     os.makedirs(directory + cartella)

if not torch.isnan(reward):

    images = []
    for epoch in range(num_epochs):
        filename = f'{directory + cartella}A_val_{epoch}.png'
        images.append(imageio.imread(filename))

    imageio.mimsave(directory+cartella+'A_val_evolution.gif', images, fps=10)

    for epoch in range(num_epochs):
        filename = f'{directory + cartella}A_val_{epoch}.png'
        os.remove(filename)