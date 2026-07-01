# Making a Rewarding RL Project

Mateus Karvat Camara

## Who is Mateus?
• Well, the TA
⁃ Email: mateus.k@queensu.ca
• PhD student under Dr. Givigi
• Took CISC 856 in 2023
• Worked on 2 Multi-agent Reinforcement Learning (MARL) research projects with Dr. Givigi
• Worked as Teaching Assistant for RL courses three times

## What is the project about?
• Well, it’s open-ended
• Possible approaches:
⁃ Replicating and contrasting existing studies (Benchmarking)
⁃ Expanding a published paper/study (Paper expansion)
⁃ Modeling a novel problem and solving it (Novel problem – talk to us about it)

## What we expect (no matter the type of project)
• Understand the algorithms being used
⁃ Especially if they are not traditional
• Look into similar approaches
• Comprehensive experimentation
⁃ Not just 2 or 3 runs, but several different experimental conditions
⁃ Avoid problems/algorithms that would take way too long to train (MARL, Vision,
training LLM/VLM/VLA)
• Depth of analysis
⁃ Tables, plots, relevant metrics
⁃ Why is X better than Y?

## What we expect (no matter the type of project)
• Project report = first version of a paper
⁃ Organization
• Follow IEEE conference paper template
• Include mathematical descriptions of the Action space, the Observation space, and the Reward
functions in your Methodology!
⁃ Rigor
• Use citations to support your claims
• As much as possible, try to quantify results/methods
• Discuss your results instead of just mentioning them

## What we expect (no matter the type of project)
• Finish the project!
⁃ Environment + Algorithms
⁃ The task does not need to be successfully completed, but we need to see significant progress and attempts at solving it
⁃ Use existing tools
• Gymnasium
• Stable Baselines 3 (don’t code your RL algorithm from scratch)
• Pygame (if relevant)
• Tensorboard or equivalent (for learning plots)
⁃ Don’t be overly ambitious!

## Benchmarking
• Read a few papers that deal with the same problem
• Make sure to understand their approach
• Papers that have open source implementations are best
• Especially if they have weights on Hugging Face
• Run them all, compare their results and maybe test them in novel scenarios

## Benchmarking - example
• Problem: Gopher
⁃ ATARI game with existing Gymnasium environment
• 5 algorithms compared:
⁃ DQN
⁃ PPO
⁃ A2C
⁃ QR-DQN
⁃ R-PPO


## Benchmarking - example
• Extensive experimentation
⁃ Unique training configurations proposed
(Table I)
⁃ Tables presenting final results (Table II)
⁃ Plots showing the results throughout
training (Fig 2, Fig 3)


## Paper Expansion
• Pick a paper and make sure to DEEPLY understand its approach
• Look for gaps and issues in their methodology
⁃ Techniques they could have used
⁃ Scenarios they didn’t explore
⁃ Subpar results
⁃ Hint: Future work section
• Replicate their experiments
⁃ If you get different results, investigate why
• Implement your improvement ideas and conduct comprehensive experiments

## Paper Expansion - example
• Paper selected:
⁃ Xu et al, 2019 - Learning an Adaptive
Learning Rate Schedule
• Gap in the methodology:
⁃ Original paper used PPO, although SAC is
a better alternative

## Paper Expansion - example
• Extensive experimentation conducted
⁃ Students were unable to replicate the
results from the original paper (PPO)
⁃ But they presented the same baseline as
the original paper: Step decay
⁃ They also added 2 new baselines: Fixed RL, Cosine

## Novel Problem
• Pick a problem you are already familiar with
⁃ The schedule for this project is tight, so a completely new problem will be challenging
• Look for current approaches to deal with this problem
⁃ Look for Gymnasium environments that might fit your problem
• Propose a way to deal with the problem using RL
• Solve the problem iteratively
⁃ Do a very limited solution to the problem first, considering several constraints
⁃ Then build from there, adding complexity at every new iteration

## Novel Problem - example
• Problem:
⁃ Tumor detection in gigapixel images
• Students were already familiar with the
problem and the literature from this field
⁃ They were able to just focus on the RL
solution for the problem

## Novel Problem - example
• Building iteratively
⁃ They did their experiments on a single
gigapixel image
⁃ There was not enough time to expand
their approach for multiple images
⁃ And that’s ok!
⁃ Their solution worked well on that image
⁃ They did several experiments on that
single image

## My CISC 856 Project
Reinforcement Learning Gait Control for Humanoid Robots

## Report
• Novel problem
    ⁃ Based on my colleague’s BSc thesis
    ⁃ Center of Mass trajectory for humanoid robots
• Several experiments conducted (over 100)
• Full PDF available as an example here on OnQ
    ⁃ This report is being provided just as an  example


Questions:
mateus.k@queensu.ca
