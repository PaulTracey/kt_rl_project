Project Setup
=============
This project uses two separate virtual environments because of dependency conflicts between CleanRL (used for MARL)
and the Pygame GUI (used for SARL).

- .venv: SARL (Single-Agent RL) with GUI
- venv_cleanrl: MARL (Multi-Agent RL) headless with CleanRL



Creating the SARL Environment
----------------------------------
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Creating the MARL Environment
-----------------------------------
python -m venv venv_cleanrl
.\venv_cleanrl\Scripts\Activate.ps1
pip install -r requirements_marl.txt


Running GUI with SARL
------------------------

1. Activate the SARL environment:
.\.venv\Scripts\Activate.ps1

2. Run the GUI:
python main.py

3. From the GUI select your algorithm from the Choose Algorithm dropdown.


Running CLI with MARL
---------------------

1. Activate the MARL environment
.\venv_cleanrl\Scripts\Activate.ps1

2. Train MARL_PPO (short example of 1k timesteps)
python -m training.train_marl_ippo --total-timesteps 1000

