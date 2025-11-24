# from gym.envs.registration import register
from gymnasium.envs import registration

registration.register(id="StagHunt-Hunt-v0", entry_point="gym_stag_hunt.envs:HuntEnv")

registration.register(id="StagHunt-Simple-v0", entry_point="gym_stag_hunt.envs:SimpleEnv")

registration.register(id="StagHunt-Harvest-v0", entry_point="gym_stag_hunt.envs:HarvestEnv")

registration.register(id="StagHunt-Escalation-v0", entry_point="gym_stag_hunt.envs:EscalationEnv")

registration.register(id="StagHunt-Hunt-PZ-v0", entry_point="gym_stag_hunt.envs:HuntPZEnv")

registration.register(id="StagHunt-Harvest-PZ-v0", entry_point="gym_stag_hunt.envs:HarvestPZEnv")

registration.register(id="StagHunt-Escalation-PZ-v0", entry_point="gym_stag_hunt.envs:EscalationPZEnv")
