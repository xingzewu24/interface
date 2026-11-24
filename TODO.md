# TODO

## Track and Metatron

- [ ] Revisit both Track ↔ Metatron transitions. Compare settling criteria, entry posture, action-history
  initialization, and blend duration in both directions. Measure joint target continuity and worst-case control tick
  time. [Policy FSM implementation](g1_playground/policy_fsm.py)
- [ ] Review where the handoff and input code live. Keep the shared transition lifecycle in one place and move
  policy-specific preparation and input mappings closer to the components that own them.
  [Policy input mappings](g1_playground/controller/policy_ctrl.py)
- [ ] Review the boundary between skills, prompts, and model encoding. Prototype alternatives to the current
  `stand`, `idle`, and `steer` behaviors. Compare prompt target frames, timing, and masks using stance stability
  and steering response in simulation.
  [Skills](g1_playground/interface/metatron/skills/skills.py) · [Prompts](g1_playground/interface/metatron/prompts/prompt.py)
- [ ] Design the split with policy inference and the control loop on Linux, and skill execution on macOS. Define the
  interface for robot state, skill commands, prompt updates, and status feedback, including timestamps and behavior
  when messages are delayed or the connection drops. Design a harness that can run both sides against MuJoCo, replay
  recorded sessions, and test timing and reconnection across the two machines.
