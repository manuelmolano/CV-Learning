#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:44:59 2021

@author: molano
"""
import numpy as np

import neurogym as ngym
from neurogym import spaces


class MakingDelayResponse_stage(ngym.TrialEnv):
    """Perceptual decision-making with delayed responses.

    Agents have to integrate two stimuli and report which one is
    larger on average after a delay.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/s41586-019-0919-7',
        'paper_name': 'Discrete attractor dynamics underlies persistent' +
        ' activity in the frontal cortex',
        'tags': ['perceptual', 'delayed response', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
                 sigma=1.0, stage=4, max_num_rep=3):
        super().__init__(dt=dt)
        self.abort = False
        self.choices = [1, 2] # right or left options
        delays = (0) # until stage 3 there are no delays
        self.cohs = np.array([51.2])*stim_scale
        # cohs specifies the amount of evidence (modulated by stim_scale)
        # cohs = How much different are right and left stimulus
        self.sigma = 0 # How much noise is applied
        # until stage 4 there is not noise
        self.stage = stage
        self.first_counts = False # first answer is not penalized
        # this only happens in stage 1
        self.max_num_rep = max_num_rep
        self.rep_counter = 0
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.,
                        'fail': 0.}

        if stage == 0:
            self.rewards = {'abort': -0.1, 'correct': +1., 'fail': +1.}
            # differences between the 2 stimuli
            self.cohs = np.array([0])*stim_scale
        elif stage == 1:
            self.first_counts = True
        elif stage == 3:
            self.cohs = np.array([51.2])*stim_scale
            delays = (0, 300, 1000)
        elif stage == 4:
            self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stim_scale
            delays = (0, 300, 1000)
            self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        if rewards:
            self.rewards.update(rewards)        
        self.timing = {'fixation': 0, 'stimulus': 300, 'delay':delays,
                       'decision': 400}
        if timing:
            self.timing.update(timing)

        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'sigma': self.sigma,
        }
        trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        # define observations
        self.set_ob([1, 0, 0], 'fixation')
        stim = self.view_ob('stimulus')
        stim[:, 0] = 1
        stim[:, 1:] = (1 - trial['coh']/100)/2
        stim[:, trial['ground_truth']] = (1 + trial['coh']/100)/2
        stim[:, 1:] +=\
            self.rng.randn(stim.shape[0], 2) * trial['sigma']

        self.set_ob([1, 0, 0], 'delay')

        self.set_groundtruth(trial['ground_truth'], 'decision')

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = self.gt_now

        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision') and action != 0:
            # allow change of mind
            new_trial = True if self.stage != 1 else action == gt
            if self.stage == 0:
                reward = 0 if self.rep_counter >= self.max_num_rep\
                    else self.rewards['correct'] # reward is 0 if it repeating more than it should
                self.count(action)
            elif action == gt:
                reward = self.rewards['correct']
                self.performance = 1
            elif action == 3 - gt:  # 3-action is the other act
                reward = self.rewards['fail']

        info = {'new_trial': new_trial, 'gt': gt}
        return self.ob_now, reward, False, info

    def count(self, action):
        '''
        counts nubmer of repetitions of action.
        '''
        action_1_minus_1 = action - 2/action
        if np.sign(self.rep_counter) == np.sign(action_1_minus_1):
            self.rep_counter += action_1_minus_1  # add to counter
        else:
            self.rep_counter = action_1_minus_1   # reset counter
